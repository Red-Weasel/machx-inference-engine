# V4.1 port — Phase 5 gate brief (5a MoE, 5b attention, 5c whole layer)

> **Disclosure.** Only 5b had criteria written before its test was run
> (`11_PHASE5B_CRITERIA`, itself written after the test was *coded*). 5a and 5c have no
> pre-written criteria; this brief is written after all three ran and passed. So this document
> is a **claims list**, not a contract, and the gate's job is twofold: verify each claim holds by
> running, AND judge whether these are the *right* claims — whether a test could pass while the
> thing it purports to check is wrong. The builder has already found one such case in 5c
> (`13_PHASE5C_RESULT`, "A vacuous check"). Assume there may be more.

## Artifacts

| | golden | engine test | result doc |
|---|---|---|---|
| 5a | `tools/ds41_reference/golden_moe.py` | `ie-ds41-moe-test` | `10_PHASE5A_RESULT` |
| 5b | `tools/ds41_reference/golden_block.py` | `ie-ds41-attn-test` | `12_PHASE5B_RESULT` |
| 5c | `tools/ds41_reference/golden_block.py` | `ie-ds41-block-test` + `ds41_hc_parity_test` | `13_PHASE5C_RESULT` |

Harness: `tools/ds41_reference/kernel.py` (six stand-ins; see `09_GOLDEN_HARNESS`).
Commits: `3f70452` .. `49cb8be`.

## Claims

**H1. The golden is DeepSeek's code, not a reimplementation.** Only six leaf primitives are
replaced; `Attention`, `Block`, `MoE`, `Gate`, `Expert` execute from the shipped `model.py`.
Verify by reading `kernel.py` and confirming nothing else in `model.py` is monkeypatched.

**H2. The stand-ins are faithful.** `dequant_fp4`/`dequant_fp8` use the decode Phase 2 proved;
`hc_split_sinkhorn` and `sparse_attn` are transcriptions of tilelang kernels that cannot run
here. `sparse_attn` was checked against a naive dense softmax (4.8e-7) including a fully-invalid
row; `hc_split_sinkhorn` against the engine's independent V4 kernel via `ds41_hc_parity_test`.
**Attack:** could a wrong `sparse_attn` (e.g. sink in the numerator, or -inf instead of -1e30 as
the running-max floor) still pass the naive-dense check? Does the naive check share the same
assumption?

**H3. Goldens are self-consistent.** `golden_moe.py` recomputes the block independently and
gets 0.0; `golden_block.py` asserts its stage decomposition equals `Attention.forward` and
`Block.forward` at `atol=0, rtol=0`.

**H4. 5a: routing is exact and the routed-expert block tracks the reference.** Indices 0/48
differ, weights 0.000000, fp16-act rel 2.0e-4, Q8_1-act rel 3.3e-3. Routing and experts are
tested *separately* with the reference's own routing fed to the experts.

**H5. 5b: attention matches stage by stage with zero new kernels.** Seven stages at 2.2e-4 to
3.1e-4. Criterion 6 of `11_PHASE5B_CRITERIA`: the diff must contain no new kernel — verify.
Criterion 4: the conjugate rotation is genuinely tested — the builder showed the two `o`
goldens differ by 0.51/0.83 in the rope channels and 0.0 in the nope channels; verify.

**H6. 5c: a whole layer matches at 2.0e-4 with every intermediate stage real.** Nine stages.
The MoE gate runs on the HOST in this test (stated in the tool's header) — judge whether that
is an acceptable scope statement or a hole.

**H7. The one new kernel is justified and guarded.** `ds41_hc_mixes`/`ds41_hc_collapse` exist
because V4.1 consumes `pre` one sublayer later than V4 (model.py:968 vs deepseek4.cpp:3399).
Verify that claim by reading both. `ds41_hc_parity_test` pins the duplicated Sinkhorn to the V4
kernel at ~1e-6. **Attack:** is 1e-5 a bar that a wrong Sinkhorn iteration count or a dropped
`+eps` would fail? Estimate or test.

**H8. Tolerances are not hiding anything.** Every stage tolerance (1e-5 for fp32 paths, 3e-3 /
5e-3 for post-GEMM) — would a structural error (transposed weight, swapped w1/w3, wrong rope
convention, dense instead of grouped `wo_a`) produce an error that clears it? The builder
asserts each would be ~100x over. Pick at least two and demonstrate with a mutant, never
modifying the repo.

## Explicitly not claimed

Layers other than 0; the compressor, indexer, engram; KV cache across steps; decode; the
residency planner; embed/head; any performance number.
