# Reproducing the showdown benchmark (public guide)

This is the public-facing reproduction guide for `scripts/bench_showdown.sh`,
the order-controlled engine-vs-llama.cpp benchmark that produces the crown
numbers in `docs/public/2026-06-beating-llamacpp-on-intel-arc.md`.

The script is already parameterized via environment variables — nothing in it
is local-only as long as you set the three paths below. The default paths
(`$HOME/.seal/models/...`, `/tmp/lcpp-master/...`) are the author's box; set
your own.

## What you need

1. **The engine, built.** `cmake --build build -j` so that
   `build/tools/ie-bench` exists. The script aborts with a clear message if it
   does not.
2. **The model GGUF.** The crown run uses
   `Qwen3.6-35B-A3B-Q4_K_M.gguf` — the exact same file is fed to *both*
   engines (this is load-bearing: a different quant or a re-quantized copy
   invalidates the comparison).
3. **A SYCL `llama-bench`.** Build llama.cpp's SYCL backend from the commit the
   crown was measured against:
   - commit: `76da2450a` (master, b9586-class)
   - build: `-DGGML_SYCL=ON -DCMAKE_CXX_COMPILER=icpx`
   - binary: `<llama.cpp>/build-sycl/bin/llama-bench`
   (For the 27B and Qwen3-Coder-30B breadth comparisons the reference backend
   is llama.cpp **Vulkan** from master `fdc3db9b6`, built `-DGGML_VULKAN=ON` —
   those are documented in `docs/benchmark_matrix_2026-06-09.md`, not this
   crown script.)

## Driver / toolchain stack the crown numbers were taken on

- GPU: Intel Arc Pro B70 (BMG-G31, Xe2-HPG)
- NEO 26.14.37833.4, IGC 2.32.7, oneAPI 2025.3 (SYCL build env), GuC 70.60.0
- IntelLLVM 2026.0.0

GPU swings ±40 tok/s between a cold and a heat-soaked run, which is exactly
why the protocol below is order-controlled rather than A-then-B.

## Run it

```sh
GGUF=/path/to/Qwen3.6-35B-A3B-Q4_K_M.gguf \
LLAMA_BENCH=/path/to/llama.cpp/build-sycl/bin/llama-bench \
ROUNDS=3 \
scripts/bench_showdown.sh            # add --tg for the decode (tg128) pass
```

Environment variables (all optional, defaults are the author's box):

| var          | meaning                                         | default |
|--------------|-------------------------------------------------|---------|
| `GGUF`       | model file, fed to both engines                 | `$HOME/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf` |
| `LLAMA_BENCH`| path to a SYCL `llama-bench` binary             | `/tmp/lcpp-master/build-sycl/bin/llama-bench` |
| `ROUNDS`     | number of alternating engine/llama pairs        | `3` |
| `--tg` (arg) | also run the decode (tg128) pass                | off |

## The order-controlled protocol (what the script enforces)

This is the credibility core; it matches
`docs/benchmark_matrix_2026-06-09.md`:

1. **One discarded JIT warmup.** The first engine run after any rebuild pays
   SYCL JIT compilation inside the timed region; the script runs and discards
   it before measuring.
2. **Alternating pairs (new-old-new).** Each round runs the engine's pp512 and
   then llama.cpp's pp512 back-to-back, so each pair sits on the same thermal
   footing. With `--tg` the same alternation is applied to decode.
3. **Same GGUF for both engines.** Non-negotiable — the script passes the one
   `$GGUF` to both.
4. **llama.cpp flags pinned:** `-ngl 99 -sm none -mg 0`, SYCL device pinned via
   `ONEAPI_DEVICE_SELECTOR=level_zero:0` / single GPU.
5. **Same hour.** Engine and llama numbers in a run come from the same session.
6. **Thermal honesty.** The script prints a caveat to treat cross-round drift
   over ~3% as thermal noise, not a real delta.

Separately, every engine change in this project must hold the perplexity gate
`./build/tools/ie-perplexity` ≤ 6.57 (production PPL is currently 6.45) — that
is the quality tripwire behind every perf number, not part of this script.

## Expected crown result (2026-06-10 ledger)

| metric | engine | llama.cpp SYCL master | delta |
|---|---:|---:|---:|
| pp512 prefill | ~1144 ± 5 | ~1064 ± 8 | **+7.6%** |
| tg128 decode (turbo) | 84.1 | 81.31 ± 0.21 | **+3.5%** |
| tg128 decode (default) | 81.0 @ PPL 6.45 | 81.31 | tie at better quality |

Your absolute numbers will differ with driver/board/thermal state; the
order-controlled *delta* is the reproducible quantity.
