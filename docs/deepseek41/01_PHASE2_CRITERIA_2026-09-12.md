# V4.1 port — Phase 2 gate criteria (written BEFORE the build)

**Phase 2 scope**, from `00_PORT_SCOPE_2026-09-12.md` §"cheapest next step": a standalone,
host-only loader that opens the shipped checkpoint through the engine's own `SafetensorsModel`,
dequantises one dense tensor and binds one routed expert, and checks both against the reference.
No GPU. No engine forward-path changes.

The tool is `tools/ds41_probe.cpp` → `ie-ds41-probe`. The reference is torch's own
`float8_e4m3fn` / `float8_e8m0fnu` / `float4_e2m1fn_x2`, run over the SAME bytes.

## Facts this phase is built on (verified 2026-09-12, not assumed)

- `config.json`: `model_type deepseek_v41`, `quantization_config.weight_block_size [32, 32]`,
  `scale_fmt ue8m0`, `expert_dtype fp4`, 40 layers, hidden 5120, 384 routed experts,
  `moe_intermediate_size 2304`.
- `model.safetensors.index.json` **exists** (7,470,294 B) and its `weight_map` has **96,085**
  entries across **48** shards. (An earlier note in this port's working log claimed the
  checkpoint ships without an index. That was wrong; the fallback written for it is reverted.)
- Phase 1 already proved, on real bytes: engine nibble table == e2m1 spec on all 16 codes;
  scales are exact powers of two; dense block is 32x32.

## Pass criteria

Each is checkable by running the tool. Any FAIL is a phase FAIL.

1. **Discovery.** `SafetensorsModel::open(dir)` returns "" and reports `shard_count() == 48`
   and `all().size() == 96085` — matching `weight_map` exactly, not approximately.

2. **Resolution is total and correct.** For every one of the 96,085 names in `weight_map`,
   `SafetensorsModel::find(name)` returns non-null AND the returned tensor came from the shard
   the index names. This is the guard on the linear-scan → hash-index rewrite: a first-wins or
   collision bug surfaces as a name resolving into the wrong shard, which a null-check alone
   would miss.

   > **AMENDED DURING THE BUILD (disclosed, not quietly).** As written, "came from the shard
   > the index names" is not checkable without adding a `shard_of()` accessor to
   > `SafetensorsModel` — public API added solely to let a test see private state. The tool
   > instead checks three things that together bound the same failure strictly harder:
   > **(a)** exhaustive *pointer identity* — for all 96,085 tensors, `find(t.name) == &t`, so a
   > name resolving to a different object of the same name is caught, which the original
   > wording would have let through; **(b)** no name appears in two shards, so the model-level
   > first-wins scan is unambiguous; **(c)** every `weight_map` name resolves. The original
   > form only detects cross-shard confusion, which this edit did not touch (`SafetensorsModel::
   > find` still scans shards linearly and was not modified). The gate should judge whether it
   > agrees this is stronger; it is recorded here precisely so that is the gate's call.

3. **Size math.** Every distinct `dtype_str` in the checkpoint is enumerated with a count. For
   every dtype the engine maps, `bytes_for(dtype, numel) == nbytes` on every tensor of that
   dtype. For the FP4 expert planes (shipped as U8 at `[N, K/2]`), the check is on the retyped
   view: `bytes_for(kFP4_E2M1, N * 2 * (K/2)) == nbytes`. This is what proves the three new
   `TypeInfo` rows are right rather than merely present.

4. **Dense dequant is EXACT.** `layers.0.attn.wq_b.weight` + `.scale`, dequantised by the tool
   (FP8 E4M3 byte → fp32, times the E8M0 scale of its 32x32 block) vs the torch reference over
   the same bytes: **max abs diff == 0.0**, and identical NaN/Inf placement. Both conversions
   are exact and the product is a single fp32 multiply, so any nonzero diff is a bug, not
   tolerance. A tolerance-based pass here does not count.

5. **Expert bind is EXACT.** `layers.0.ffn.experts.0.w1.weight` + `.scale`, decoded with the
   engine's own nibble constant `0xC8643210` (compiled from C++, not transcribed) and the E8M0
   scale per 32 along K, vs the torch reference: **max abs diff == 0.0** over all 2304x5120
   elements. Plus: the decoded low nibble is element `2b`, the high nibble element `2b+1`
   (order, not just magnitude), verified by an asymmetric spot check.

6. **No collateral damage.** `git diff --stat` touches only `include/ie/dtype.hpp`,
   `src/core/dtype.cpp`, `include/ie/safetensors.hpp`, `src/loaders/safetensors_reader.cpp`,
   `tests/safetensors_reader_test.cpp`, `tools/CMakeLists.txt`, plus the new tool and docs.
   `safetensors_reader_test` passes. The host-only unit tests that were green before are green.

7. **Bounded cost.** The tool mmaps only; peak RSS stays under 8 GiB against a 476 GiB
   checkpoint, and it finishes without reading the engram tables (188.8 GiB) into memory.

## Explicitly NOT in this phase

Config reader, tensor binder, CED forward, CSA2, the hierarchical indexer, the NVMe expert
tier, engram, FP4 KV, ViT, DSpark. Phase 2 answers one question only: *can this engine read
these bytes and get the same numbers as the reference?*
