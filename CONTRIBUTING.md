# Contributing

Thanks for looking. MachX is an inference engine for Intel Arc GPUs: SYCL kernels, GGUF model
loading, and an OpenAI-compatible server.

## What you need

- **Intel oneAPI 2026.x** (the SYCL compiler)
- **An Intel Arc GPU.** Most of this code only runs on one. Kernel work cannot be verified
  without hardware.
- CMake and Ninja

If you don't have an Arc card, the Docker path in [QUICKSTART.md](QUICKSTART.md) still lets you
run a model, and documentation or host-side fixes are still reviewable.

## Build

```bash
source scripts/env.sh
cmake -S . -B build -G Ninja && cmake --build build -j
./build/src/ie pull llama8b
./build/src/ie serve <model.gguf> --gpus 1
```

The build is large. Linking wants real memory free — building while a big model is loaded can
take the machine down.

## Tests

Test sources are under `tests/` (`tests/unit/` for kernels, `tests/integration/` for the
loading and serving paths). They build as their own executables alongside the engine; run the
ones your change touches from `build/`, for example:

```bash
./build/tests/attention_test
```

A kernel change should come with a numerical check against a reference — see the existing
tests for the pattern, and `scripts/gen_*_golden.py` for how goldens are produced.

## What a good change looks like

- **Measure performance claims.** "Faster" needs a before and after on a stated model, context
  length and GPU, from the same machine. Numbers from different conditions are not comparable.
  A change that helps a small model and not a large one is usually not worth its complexity
  here.
- **Prove a kernel is still correct.** Bit-exact against the previous output where the change
  should be lossless; a perplexity comparison where it is not.
- **Surgical.** Match the style around you, including comment density. Comments in this
  codebase carry the reason and the evidence — the measurement, the failure, the date — not a
  restatement of the code.
- **Watch for SYCL traps.** Two that have cost real days here: two identically-named kernel
  helpers in an anonymous namespace across translation units make the wrong device code run,
  so prefix every kernel helper; and a pageable-vector host-to-device copy can stall for
  minutes where pinned host memory does not.
- **State what you could not verify.** Hardware differs. "Tested on A770, not on B-series" is
  useful; implying broader coverage is not.

## Reporting a bug

Include your GPU, driver and oneAPI versions, the model file, and the exact command. Engine
logs go to stderr — the `[gen]` lines carry the prefill and decode timings that usually
identify the phase at fault.

## Security

Don't open a public issue for a vulnerability. Use GitHub's private vulnerability reporting on
this repository instead.
