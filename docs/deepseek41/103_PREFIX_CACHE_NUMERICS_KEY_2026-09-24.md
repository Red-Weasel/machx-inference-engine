# V4.1 — #48: the disk prompt cache is keyed by its numerics, not by the executable

2026-09-24, branch `engine-fix-48-29-54`. **Status: code + CPU tests; the GPU check is in the runbook**
(`docs/RUNBOOK_2026-09-24_fixes-48-29-54.md`, section C).

## Why

Phase 47 (docs/87) keyed every disk entry by the executable's size and mtime. The reason is sound: an entry computed by
different arithmetic must not be loaded. But the key was too coarse. Every rebuild, even of an unrelated file, emptied the
cache ("0 entries for this build") and cost a ~30 s cold first turn (fix list #48).

## The key now

    <model> | format 1 | numerics <16 hex> | runtime <oneDNN x.y.z hash> | <card 0 name> driver <v>; <card 1 name> driver <v>

* **model**: unchanged. The layer count, dims, window, per-layer ratios and kv-source flags, and a fingerprint of the first
  MiB of the embedding.
* **format**: `kDs41PrefixDiskFormat` (1). It is the file header's `ver` field, now one named constant. Bump it when
  `prefix_persist`'s layout changes.
* **numerics**: FNV-1a 64 of the **numerics manifest**, a text the build writes into
  `build/src/generated/deepseek41_numerics_manifest.h`. It holds:
  * the compiler's own `--version` line;
  * the flags: build type, `CMAKE_CXX_FLAGS` (the general ones and the build type's), the SYCL target, the device hint,
    `ie_sycl_flags`' compile and link options and definitions, oneDNN on/off and the version found at configure time,
    the ESIMD switch, and the per-source options of the listed files (`-fopenmp`, `-mavx2 -mfma`, `-fp-model=precise`);
  * the SHA-256 of every listed file.
* **runtime**: what still generates arithmetic after the build. The oneDNN library the process actually loaded
  (`onednn_runtime_version()`) matters because its GPU kernels are generated at run time: a moved
  `/opt/intel/oneapi/dnnl/latest` link changes them under an unchanged binary. Each card's name and driver version
  matters because spir64 kernels are JIT-compiled by the driver. The old key had neither.

`include/ie/deepseek41_prefix_key.hpp` derives the key. It is pure C++, so it has a CPU unit test.

## What is hashed: `src/model/deepseek41_numerics_inputs.txt`

67 files at first (70 since 2026-09-25, below): 28 sources, 37 headers and 2 vendored headers. The file gives the reason
for every entry. It also explains the five exclusions:

* the prompt renderer and tokenizer: they make the ids, and the ids are the lookup key;
* the drafter: speculation keeps the cache off;
* the residency planner: only its unit test calls it;
* `ds41_engine.cpp`: it sets options, not arithmetic;
* the allocator.

The rule for doubt was to include a file. For example, `src/model/deepseek4.cpp` is listed, all 320 KB of it, because it
holds `ds4_hc_mix`, and `src/core/expert_stream.cpp` is listed because it holds the slot views. The cost is a spurious
invalidation: one cold first turn after a change to V4 or to the expert streaming code.

`tests/unit/ds41_prefix_key_test.cpp` fails if a listed file includes a header that is not listed. So a new `#include` in
a numerics source cannot slip past the key.

**2026-09-25 (branch `engine-followups-0925`): the #include check was not enough.** The gate of batch 2 ran `nm` over the
listed files' objects and found calls into unlisted sources: `ds4_vision.cpp` calls `gemm_fp16` (`src/ops/gemm_fp16.cpp`)
for the vision tower, so an edit there could change an image inside a saved prefix without changing the key. The same
check found `src/model/qwen4_image.cpp`, which compiles the stb_image implementation (`STB_IMAGE_IMPLEMENTATION` and its
`STBI_ONLY_*` set) that decodes the pixels; an image's positions are keyed by its file bytes (`ds41_image_id` in
`ds41_engine.cpp`), not by its pixels. Both are listed now, with `include/ie/qwen4_vision.hpp` (the closure): **70 files**
(30 sources, 38 headers, 2 vendored). Fingerprint on this build's flags: `56c47f9a307993de` (67 files, 3f00599) ->
`47a4b6589ccc0224` (70 files, this branch, which also edits the listed `deepseek41_experts.cpp`) -> `e1fed142d96f7912`
(after the second teardown-drain wrap in the same file).

The other unlisted sources the listed objects call are now named on `unlisted:` lines of the input list, each with its
reason: V4-only code in the listed `deepseek4.cpp` (`deepseek4_cache.cpp`, `ds4_decode_gemv.cpp`, `gguf_reader.cpp`,
`dtype.cpp`), IQ3_XXS banks (`gemv_iq3_xxs.cpp`, V4 GGUF only), and `elementwise.cpp` (`cast_fp32_to_fp16`) plus
`gemv_mxfp4.cpp`, which V4.1 reaches only through `ds4_experts_forward` in the NON-resident streaming-MoE branch of
`deepseek41_forward.cpp` -- `set_prefix_cache` refuses non-resident mode, so no entry is written there.
`ds41_prefix_key_test` part 3 makes this automatic: ctest passes it `$<TARGET_OBJECTS:ie_core>`, it runs `nm` over them,
and it fails when a listed source calls a function (or reads a global) that no listed object defines and whose defining
source is not named `unlisted:`.

## Build mechanics (`src/CMakeLists.txt`, `src/model/deepseek41_numerics.cmake`)

* At configure time the flags are written to `build/src/generated/deepseek41_numerics_flags.txt`. The file is rewritten
  only when its content changes.
* At build time one custom command runs the script. Its output is a stamp; the header is a byproduct. Ninja gets
  `restat`, and the command's inputs are the listed files, the list, the flags file, the script and the compiler binary.
  The header is rewritten only when its content changes, so a `git checkout` or a `touch` that leaves the bytes the same
  recompiles nothing.
* The compiler line comes from running the compiler at build time, not from CMake's cached value. An in-place compiler
  upgrade is seen at once: the binary is an input.

Checked on a scratch configure of this branch, building only the manifest target (no compilation):

| change | result |
|---|---|
| nothing | `ninja: no work to do` |
| an unlisted source touched (`src/model/qwen36.cpp`) | `ninja: no work to do` |
| a listed source touched, bytes unchanged | the script runs; the header is not rewritten |
| a listed source's bytes changed / a flag changed | new fingerprint (checked on a scratch copy of the inputs) |
| a listed file missing | the build stops with the file's name |

## What the owner will see

* **First start on this build**: every existing entry carries an old-format key, so it is ignored. The start line says
  `disk <dir> (0 entries for numerics <fp>; N written under another key ignored)`. The 8 GiB LRU budget ages the old
  files out; nothing deletes them eagerly. One cold first turn follows.
* **A rebuild that touches none of the listed files (70 since 2026-09-25) keeps the entries.** A change to any of them does not, and that includes
  a comment. Such an invalidation is spurious, but it is the safe direction.
* Two trees with the same sources and flags (the main tree and a worktree) share entries, because the paths in the
  manifest are relative to the repository.

## Known limits

These are not changed here. The old key had the same gaps.

* **Runtime knobs that change arithmetic are not in the key**: `IE_DS41_PREFILL_XMX`, `IE_DS4_EXPERT_XMX`,
  `IE_DS41_CPU_MISS`, `IE_DS41_CPU_CONT_ROWS`, `IE_DS41_QSTAR_CONT`, `IE_DS41_CONT_GATHER`, `IE_DS41_CAND`,
  `IE_DS41_MAX_FWD` (the chunk length) and `IE_ONEDNN_DETERMINISTIC`. A process started with a non-default value loads
  entries written under the default.
* **A prefill was never bit-reproducible across processes anyway.** The CPU expert leg serves pinned-tier misses of
  prefill chunks too (`cont_rows` 2048, on since 09-21). Which pinned experts miss depends on the stream cache's history.
  The CPU leg (fp32 activations) and the XMX route (fp16 activations) round differently. So an entry is "state this code
  computed for these ids", not "the bits a fresh prefill would produce here". The key keeps that guarantee and no more.
* Unverified: arithmetic that a change outside the list can still alter is not detected. Two examples: a kernel-name
  collision across translation units (the docs/deepseek4/70 trap), and a system library other than oneDNN and the driver.
* The call check (test part 3, 2026-09-25) sees link-time symbol references from the LISTED objects only. It does not
  see: a call from an unlisted file into numerics (`ds41_engine.cpp` is unlisted by argument, not by check); an
  indirect call through a pointer, `std::function` or virtual whose target is set up outside the listed files; and a
  function-level path (it is object-level: a listed object that calls into an unlisted file from V4-only code looks the
  same as one that calls from the V4.1 path, which is why every `unlisted:` line carries a reason someone read).
  Naming a file `unlisted:` is a human claim; the test only stops a NEW unlisted callee from arriving unnoticed.
* An entry written by a larger-context process loads in a smaller one only if its latent counts fit (the existing check
  in `pc_load_disk`).

Noticed, not changed: `ds41pc_fnv` in `deepseek41_prefix_cache.cpp` seeds with `1469598103934665603`, which is FNV's
offset basis without its last digit. It is deterministic and only names files and fingerprints the embedding, so it is
harmless. The new `ds41_fnv1a64` uses the standard constant, and the unit test checks it against the published vectors.

## Tests

* `ds41_prefix_key_test`: 32/32 on the CPU. It checks FNV-1a against the published vectors; the same inputs give the same
  key; a changed build type, device hint, oneDNN switch, per-source option, file hash, added file, compiler, model,
  format, oneDNN runtime or driver each give another key; and the list's closure under `#include`. Three mutations of the
  list are caught: a header dropped, a vendored header dropped, and a missing file plus a duplicate.
  2026-09-25: 37/37 under ctest, which now adds part 3 (the `nm` call check over ie_core's 130 objects, 251 calls; 32
  without `--objects`). A list without `gemm_fp16.cpp`, `qwen4_image.cpp` and the `unlisted: elementwise.cpp` line fails
  it, naming `gemm_fp16` <- `ds4_vision.cpp`, the `stbi_*` calls <- `ds4_vision.cpp` / `ds41_vision.cpp`, and
  `cast_fp32_to_fp16` / `cast_fp16_to_fp32` <- `deepseek4_experts.cpp` / `deepseek4.cpp`.
* The GPU steps are the runbook's section C: the existing `ie-ds41-cache-test`; an entry surviving an unrelated rebuild; an
  entry not loaded after a listed file changes; and the entry loadable again once the change is reverted.
