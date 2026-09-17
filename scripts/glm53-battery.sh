#!/usr/bin/env bash
# glm53-battery — post-reboot measurement battery (2026-08-29 campaign).
# Every number this boot-cycle before a reboot was driver-rot-poisoned
# (ccs+bcs engine resets 05:09); run this ONLY on a healthy boot.
# Sequential guarded runs — the single-flight lock keeps them one at a time.
#
# Full outputs land in results/glm53-battery-<ts>/<step>.log and the
# generated text of each decode run in <step>.txt; text equality is judged
# HERE (the 08-29 "divergence" false alarm came from comparing truncated
# log tails by eye — never again).
set -uo pipefail
# Pin the repo root ABSOLUTELY before snapshotting — the snapshot copy lives
# in $OUT, so recomputing the root from the copy's own path lands in
# results/ and every relative path breaks (caught by the first real run).
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
M="$HOME/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf"
RUN="scripts/ie-run-guarded --timeout 3600"
BIN=./build/tools/ie-glm5next-run
WIKI="$HOME/llama.cpp/wikitext-2-raw/wiki.test.raw"
OUT="${OUT:-$ROOT/results/glm53-battery-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
# Execute from a snapshot: editing a running bash script misparses it
# (self-inflicted during the 08-29 shakedown). The copy also records
# exactly what ran alongside its results.
if [ -z "${BATTERY_SNAPSHOT:-}" ]; then
    cp "$0" "$OUT/battery-as-run.sh"
    BATTERY_SNAPSHOT=1 ROOT="$ROOT" OUT="$OUT" exec bash "$OUT/battery-as-run.sh" "$@"
fi
log() { echo; echo "════ $* ════"; }

# run <step> <cmd...>: tee full output; extract the generated text (from the
# [gen] prompt line to [gen] done, exclusive) into <step>.txt.
run() {
    local step="$1"; shift
    "$@" 2>&1 | tee "$OUT/$step.log"
    # engine markers ("[glm5next] ...") interleave mid-token-stream and split
    # lines — excise marker+its newline (sed -z) to restore the raw stream.
    awk '/^\[gen\] prompt:/{f=1;next} /^\[gen\] done:/{f=0} f' "$OUT/$step.log" \
        | sed -z 's/\[glm5next\][^\n]*\n//g' > "$OUT/$step.txt" || true
}
# cmp_text <a> <b>: PASS iff the generated texts are byte-identical AND
# non-empty (two empty extractions passed vacuously on the broken first run).
cmp_text() {
    if [ ! -s "$OUT/$1.txt" ] || [ ! -s "$OUT/$2.txt" ]; then
        echo "TEXT $1 vs $2 : INVALID (empty extraction — run failed?)"
        return
    fi
    if cmp -s "$OUT/$1.txt" "$OUT/$2.txt"; then
        echo "TEXT $1 == $2 : PASS"
    else
        echo "TEXT $1 != $2 : FAIL (diff follows)"
        diff "$OUT/$1.txt" "$OUT/$2.txt" | head -6
    fi
}

log "0. driver health gate"
# grep -c (not -q): -q's early exit SIGPIPEs journalctl and pipefail then
# reports the pipeline FAILED on a match — the gate silently passed on a
# rotten boot (caught by the 08-29 shakedown).
if [ "$(journalctl -k -b --no-pager 2>/dev/null | grep -ci 'engine reset')" -gt 0 ]; then
    if [ "${IE_BATTERY_FORCE:-0}" = "1" ]; then
        echo "!!! SHAKEDOWN MODE: xe resets present — ALL SPEED NUMBERS INVALID !!!"
    else
        echo "xe engine resets found in THIS boot — numbers would be garbage. Reboot first."
        exit 1
    fi
fi

log "1. warm page cache (all shards)"
for f in "$HOME/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/"*.gguf; do
    dd if="$f" of=/dev/null bs=16M status=none &
done
wait

log "2a. decode 96 tok — pooled memcpy staging (IE_G5_PREAD=0)"
IE_G5_PREAD=0 run 2a $RUN $BIN "$M" --gpus 2 --ngen 96
log "2b. decode 96 tok — pread staging (IE_G5_PREAD=1, current default)"
run 2b $RUN $BIN "$M" --gpus 2 --ngen 96
cmp_text 2a 2b   # staging must not change tokens; winner becomes default

log "3. decode 96 tok — grouped MoE kernels"
IE_G5_GROUPED=1 run 3 $RUN $BIN "$M" --gpus 2 --ngen 96
cmp_text 2b 3    # bit-identical contract

log "3b. decode 96 tok — R9 prefetcher (76.8% proxy)"
IE_G5_PREFETCH=1 run 3b $RUN $BIN "$M" --gpus 2 --ngen 96
cmp_text 2b 3b   # caching must not change tokens

log "3c. the 08-29 corrupting geometry retest (ECACHE 25600, was '!!!'-collapse)"
IE_G5_ECACHE_MB=25600 run 3c $RUN $BIN "$M" --gpus 2 --ngen 96
cmp_text 2b 3c   # PASS => collapse was boot-rot; FAIL => latent engine bug
                 # -> bisect with IE_G5_UNIFORM_SLOTS / IE_G5_ECACHE_PAD

log "4. EP decode 96 tok (async worker's first honest measurement)"
IE_G5_EP=1 IE_G5_EP_DECODE=1 run 4 $RUN $BIN "$M" --gpus 2 --ngen 96
log "4b. EP + grouped + prefetch decode 96 tok (the full stack)"
IE_G5_GROUPED=1 IE_G5_PREFETCH=1 IE_G5_EP=1 IE_G5_EP_DECODE=1 run 4b $RUN $BIN "$M" --gpus 2 --ngen 96
cmp_text 4 4b    # EP add-order differs from non-EP (2b); 4 vs 4b must match

log "5. EP oracle arms (healthy-driver verdict on the wobble class)"
log "5a. control arm (grouped OFF, prefetch OFF)"
IE_G5_EP=1 IE_G5_EP_DECODE=1 IE_G5_EP_VERIFY=1 run 5a $RUN $BIN "$M" --gpus 2 --ngen 12
grep "epv" "$OUT/5a.log" | grep -v "max|d| 0.000000 mean" || echo "  all checks ZERO"
log "5b. grouped arm"
IE_G5_GROUPED=1 IE_G5_EP=1 IE_G5_EP_DECODE=1 IE_G5_EP_VERIFY=1 run 5b $RUN $BIN "$M" --gpus 2 --ngen 12
grep "epv" "$OUT/5b.log" | grep -v "max|d| 0.000000 mean" || echo "  all checks ZERO"
log "5c. prefetch arm (peer-side pf) — wobbles here but not 5a => pf race"
IE_G5_PREFETCH=1 IE_G5_EP=1 IE_G5_EP_DECODE=1 IE_G5_EP_VERIFY=1 run 5c $RUN $BIN "$M" --gpus 2 --ngen 12
grep "epv" "$OUT/5c.log" | grep -v "max|d| 0.000000 mean" || echo "  all checks ZERO"

log "6a. prefill ppbench chunk 1024 — serial"
run 6a $RUN $BIN "$M" --gpus 2 --ppbench 2 --chunk 1024 --ppl "$WIKI"
log "6b. prefill ppbench chunk 1024 — PIPELINED (the pp>100 candidate)"
run 6b $RUN $BIN "$M" --gpus 2 --ppbench 2 --chunk 1024 --ppl "$WIKI" --pipeline
log "6c. prefill ppbench chunk 2048 — single max-context chunk"
run 6c $RUN $BIN "$M" --gpus 2 --ppbench 1 --chunk 2048 --ppl "$WIKI" --pipeline

log "6d. prefill LAYER-STREAMING (IE_G5_PP_STREAM; needs small ecache) — the pp>100 candidate"
IE_G5_PP_STREAM=1 IE_G5_ECACHE_MB=12000 run 6d $RUN $BIN "$M" --gpus 2 --ppbench 2 --chunk 1024 --ppl "$WIKI" --pipeline

log "6e. layer-streaming PPL stamp (residency change must not move numerics)"
IE_G5_PP_STREAM=1 IE_G5_ECACHE_MB=12000 run 6e $RUN $BIN "$M" --gpus 2 --ppl "$WIKI" --chunk 512

log "7. PPL band gate (chunk 512; band 1.868-1.873)"
run 7 $RUN $BIN "$M" --gpus 2 --ppl "$WIKI" --chunk 512

log "8. prefetch depth A/B (depth-1 61.8% consumed vs depth-2 49.2% on rot-boot)"
IE_G5_PREFETCH=2 run 8 $RUN $BIN "$M" --gpus 2 --ngen 96
cmp_text 3b 8

log "9. decode champion variance (pf-d2 x2 back-to-back)"
IE_G5_PREFETCH=2 run 9a $RUN $BIN "$M" --gpus 2 --ngen 96
IE_G5_PREFETCH=2 run 9b $RUN $BIN "$M" --gpus 2 --ngen 96
cmp_text 9a 9b

log "SUMMARY (tok/s lines)"
grep -h "gen. done\|TOTAL\|ppl" "$OUT"/*.log 2>/dev/null | sort -u
echo "full outputs: $OUT"
log "battery complete"
