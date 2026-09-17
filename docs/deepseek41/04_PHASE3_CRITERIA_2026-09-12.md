# V4.1 port — Phase 3 gate criteria (written BEFORE the build)

**Scope:** a config reader and a tensor binder for `deepseek_v41`, plus a host-only load test.
No GPU, no device memory, no forward pass. Phase 2 proved the bytes decode; Phase 3 proves the
*structure* — that every one of the 96,085 tensors has a known role and the predicted shape.

Artifacts: `include/ie/deepseek41.hpp`, `src/model/deepseek41.cpp`,
`tools/ds41_load_test.cpp` → `ie-ds41-load-test`.

## Facts this phase is built on (read from the checkpoint and the shipped reference, not assumed)

- `Transformer.forward` is a flat loop over 40 identical `Block`s — no encoder/decoder split
  (see `03_ARCH_FROM_THE_CODE`).
- `compress_ratios` = 2 zeros, 18 twos, 20 ones, then 3 zeros for MTP.
- `kv_source_layer_ids = [2, 8, 14, 20]`, `index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]`.
- `compressor.wgate` exists on layers 2, 8, 14 but **not** 20, because `Compressor.__init__`
  only builds a gate when `compress_ratio > 1` and layer 20's ratio is 1.
- Dense FP8 weights carry a **32x32** block scale grid; routed FP4 experts carry a **1x32
  along K** scale grid. Two different geometries, both verified against shipped shapes.
- `wo_a` is **F8_E4M3 in the shipped checkpoint**, not bf16. The reference's
  `dtype=torch.bfloat16` argument describes the unquantised model; the fp8 checkpoint overrides
  it. (`03_ARCH_FROM_THE_CODE` says bf16 — that line is wrong and is corrected by this phase.)
- Safetensors shapes are `[out, in]`.

## Pass criteria

1. **Config parse is total and faithful.** Every field the forward pass needs is read from
   `config.json`. The test re-reads the JSON independently and compares field by field; any
   required field that is absent is a hard error naming it, **never a silent default**. A
   deliberately corrupted config (missing `n_routed_experts`) must fail the load with a message
   naming that key.

2. **Per-layer kind is derived, not hardcoded.** For all 43 layers, `compress_ratio`,
   `is_kv_source`, `is_index_source`, `has_gate`, `has_engram`, `n_routed`/`n_activated` come
   from the config arrays. The derived kinds must reproduce the shipped schedule exactly:
   compressors on {2,8,14,20}, gates on {2,8,14}, indexers on {2,8,14,20,24,28,32,36},
   engram on {1,14}, and MTP layers 40-42 at 128 routed / 3 activated.

3. **Binding is total, both directions.**
   (a) Every role the kind says must be present is bound non-null; any absence is a hard error.
   (b) Every role the kind says must be ABSENT is null; a present-but-forbidden tensor is
       equally a hard error.
   (c) **Zero unclaimed tensors**: the tool accounts for all 96,085 by name — bound to a role,
       or explicitly classified as a deferred group (vision, MTP, engram tables). A name that
       matches nothing is reported and fails the phase. This is the criterion that catches a
       module I did not know existed.

4. **Shape validation on every bound tensor.** Each bound weight's shape is checked against
   what the config predicts (e.g. `wq_b` is `[n_heads*head_dim, q_lora_rank]` =
   `[32768, 1280]`). A mismatch is a hard error naming the tensor, the expected shape and the
   found shape. Deliberately corrupting one expected shape in the test must produce that error.

5. **Scale planes are paired and their geometry checked.** Every FP8/FP4 weight is bound
   together with its `.scale` sibling, and the scale grid must tile the weight exactly:
   32x32 for dense, 1x32-along-K for routed experts. A weight with a missing or
   wrongly-shaped scale is a hard error.

6. **Residency accounting reproduces the census.** The model reports bytes by group, and the
   totals equal the independently measured census (`scratchpad/budget.py`) **to the byte**:
   routed experts 268.95 GiB / 92,160 tensors, engram 189.13 GiB / 12, total 475.24 GiB /
   96,085.

7. **Host-only and bounded.** No GPU, no device allocation. Under 60 s and under 8 GiB peak
   RSS against the 476 GiB checkpoint.

8. **No regression.** The host-only unit tests green at the end of Phase 2 are still green, and
   nothing outside the new files plus `src/CMakeLists.txt` / `tools/CMakeLists.txt` changes.

## Explicitly NOT in this phase

Device upload, expert tiering, the NVMe tier, any kernel, any forward pass, engram lookup,
vision, DSpark. Phase 3 answers: *does every byte in this checkpoint have a known home?*
