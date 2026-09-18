# V4.1 — Phase 55: the vision tower

## What the checkpoint ships

`inference/vision.py` + `image_processor.py`, and 309 tensors (926 MiB BF16) the engine loaded and set aside until now:
a 32-block ViT (1024 wide, 16 heads, 2-D RoPE over the whole 64-d head, SwiGLU 2816, RMSNorm), a 3x3-unfold
aligner into the 5120-wide LM (`aligner.w1` 9216 -> 5120, GELU, `w2`), three learned delimiter rows
(`image_start`, `image_newline`, `image_end`) and a routing bias for image-span tokens in every MoE gate
(`ffn.gate.bias_vl`). An image of w x h px is contained + grey-padded onto a canvas of 14 px patches, at least
544 x 544 px, at most 1,024 LM tokens; it enters the LM as

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

It is the V4 Vision-Exp tower (docs/deepseek4/70) at other sizes: a 5120-wide output instead of 4096, a 1,024-token
budget instead of 384, no pad row and no compress-pad alignment.

## What was built

- `Ds4Vision` (src/model/ds4_vision.cpp) generalised, the V4 sidecar path unchanged: `load_from(find, options)` takes
  tensors from any source (the checkpoint's shards); `Ds4VisionOptions` sets the output width, the scratch caps and:
  - **transient mode**: the converted f16 weights live in pinned host memory (staged once, 0.4-1.4 s); `encode_gpu()`
    leases ONE device block (weights + scratch, 1,487 MiB) for the encode and returns it. The V4.1 cards are full at
    steady state (the expert pool is sized to the free VRAM), and the encode runs just before the prompt's prefill.
  - **tiled attention**: 2,304 query rows per scores GEMM (softmax is per row, so the tiling is exact); a 1,024-token
    image is 9,189 patches, and a whole scores matrix would be 9189^2 x 4 B = 322 MiB per head.
  - **scaled probabilities**: the probability matrix is an f16 GEMM input; over 9,000 keys most entries fall below
    f16's normal range and lose their low bits. Stored as p x 256 (the scatter divides it out). Scale 1 is the old
    arithmetic bit for bit (V4 keeps it).
- `include/ie/ds41_vision.hpp` / `src/model/ds41_vision.cpp`: the plan (`plan_image_grid`, `safe_resize`,
  `solve_resize_ratio`), the span layout, the loader on the V4 PIL-exact resampler.
- `tools/ds41_reference/golden_vision.py`: the shipped modules on the shipped weights, fp32 on the CPU, for three
  fixtures (text + shapes 640 x 480, a 1920 x 1080 screenshot at the 1,024-token budget, a tall 200 x 900 image below the
  pixel floor) and a 156-size plan sweep.
- `tools/ds41_vision_test.cpp` (`ie-ds41-vision-test`): the gate.

## Gate: 27/27 (`~/ds41_work/p55/vision_test_final2.log`)

| check | result |
|---|---|
| plan vs `plan_image_grid`, 156 sizes | 0 mismatches (largest grid 9,189 patches, cap 9,216) |
| loader pixels vs PIL | 0 samples beyond 2 u8 LSB on all three fixtures |
| tower rows vs fp32 reference, fixture 0 | rel-L2 2.8e-3, min row-cos 0.99984, p05 0.99998 |
| fixture 1 (968 tokens) | rel-L2 6.1e-3, min row-cos 0.9786, p05 0.99997, 1 of 943 rows below 0.99 |
| fixture 2 | rel-L2 1.1e-3, min row-cos 0.99999 |
| run to run | bit-identical |
| transient block returned after every encode | yes (70 MiB of one-time runtime state after the first) |
| span layout + delimiter rows | exact |
| engram hash with image positions (Phase 56) | 0 mismatches vs `NgramHashState` under its token mask |

Encode: 211 ms for a 1,610-patch image, 2.76 s at the 9,189-patch budget (attention-bound: 16 heads x 4 tiles).

**The bar, and why it is not "every row above 0.99".** The yardstick is the shipped module in its own precision:
the official tower run in bf16 against its fp32 output (`~/ds41_work/p55/bf16_yardstick.py`) measures rel-L2 5.1e-2,
min row-cos 0.607 and 26 of 943 rows below 0.99 on fixture 1 -- the rows over the flat dark background are small and
precision-sensitive. The engine's f16 tower is 8x closer to the exact function than the model's own bf16 run on
every fixture. The gate: rel-L2 < 2e-2, p05 row-cos > 0.999, at most 0.5 % of rows below 0.99, none below 0.95.

## Found on the way

- **icpx's fast FP model broke the plan.** `std::floor(height * beta / 14)` came out one patch short on 6 of 156
  sizes (`x / 14` compiled as `x * (1/14)`). `src/model/ds41_vision.cpp` is built with `-fp-model=precise`.
- **f16 probabilities.** At scale 1 the 968-token fixture's worst row was 0.9688; at 256 it is 0.9786 and rel-L2
  6.95e-3 -> 6.09e-3. `IE_DS4_VIS_PSCALE` overrides it (A/B lever).
