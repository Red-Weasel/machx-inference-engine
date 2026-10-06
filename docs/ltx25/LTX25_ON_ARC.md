# LTX-2.5 on two Arc Pro B70 cards

Measured October 5–6, 2026 on the two-card test machine (Arc Pro B70, 32 GB each; PyTorch 2.14.1+xpu, diffusers from its
main branch). The video scripts are separate from this repository for now; this page records what was measured and how.
Unlike the language-model engine, the video pipeline runs in Python on PyTorch's Intel GPU build, with Mach X kernels
(`mxv_*`, C++/SYCL, oneDNN underneath) called on PyTorch's own queue.

## Result

Five seconds of 1280×704 video with sound (121 frames, 24 fps) from Lightricks' distilled LTX-2.5:

| version | one clip, generate | what changed |
|---|---:|---|
| stock PyTorch, the two cards used one after the other | 79 s | the reference (7.1 s a step, 8 steps, 11 s decode) |
| both cards working in every block, 8-bit weights | 53 s | video tokens split between the cards (4.79 s a step) |
| + Mach X multiply, the model card's two-stage flow, connectors once, pixel and sound decode on different cards | 26.8–26.9 s warm (29.9 s first clip) | |
| + fused norm and rotary kernels, in-place K/V, decode tiles on both cards, exact partial attention for the sound | **20.9 s warm** | 3.8× the stock run |

- **One card alone:** 33.8 s. **One process per card**, each making its own clips: both run at 34 s with no slowdown, so a clip
  every **17 s** (about 210 an hour). Six different prompts took 160 s of wall time including loading both processes.
- **Loading:** weights on both cards and prompts encoded in 20 s (67 s before the load work). A process that makes one clip:
  20 s load plus about 29 s for the first clip (the first clip pays about 8 s of first-decode allocations).
- Frames and sound tracks were inspected on the stock and 26.8 s clips (coherent, sound present); every clip is also checked
  automatically (see Known issues).

## The model

Lightricks LTX-2.5: a 48-block transformer, 4096 wide, 32 heads × 128, with a video and an audio stream in every block (38 GB
in bf16); Gemma 4 12B as the text encoder (24 GB, run once per prompt and freed). The distilled schedule runs 8 steps with
one pass per step. A 5 s clip at 1280×704 is 14,080 video tokens and 126 audio tokens per block.

## How it got from 79 s to 21 s

1. **Both cards in every block.** The transformer is stored as int8 plus one bf16 scale per 32 values (the GGUF Q8_0 layout,
   about 20 GB a card instead of 38), so each card holds all 48 blocks. Card 0 carries the first half of the video tokens, card 1
   the second, the sound tokens run on both; the video self-attention's K/V and the sound-attends-to-video K/V cross through
   pinned host memory (a direct card-to-card copy of bf16 corrupted data in this PyTorch build, so none is used).
2. **Mach X video kernel #1, `mxv_linear_q8`.** The Q8_0 weights go straight into the XMX multiply (oneDNN's generated gemm) with
   bf16 activations: no expand pass, a step from 4.77 to 4.46 s, and 4.6× more accurate than the cast it replaces (0.4 % against
   1.7 % from a float32 product).
3. **The model card's two-stage flow:** 8 steps at half size (640×352), a 2× latent upscale, 3 steps at full size. The largest
   single gain. Connectors are computed once per prompt and freed (6.3 GB that used to be uploaded before every run).
4. **Decode:** pixels in spatial tiles on card 1 with the sound on card 0 at the same time (11 s → 5.1 s), then tiles on both
   cards (→ 3.1 s, bit-identical).
5. **Fused norm and rotary kernels** (full-size step 4.46 → 3.85 s), **K/V built in place** with early hand-over (−3 ms a block),
   and **sound attends to video through exact partial attention** (log-sum-exp combine), which saves moving 58 MB of K/V
   (−1.9 ms a block).

Where a full-size block goes now (76 ms a card): the video self-attention core 17.5 ms (PyTorch's runs at 89 TFLOP/s against about
150 for plain multiplies), the eight 4096-wide linears and the feed-forward about 28 ms (105–141 TFLOP/s, 70–95 % of the measured
multiply peak), the K/V exchange about 13 ms (the card-to-card link, 5.4 GB/s a direction, limits splitting one clip across two
cards; one clip per card avoids it), norms, rotary and residuals about 8 ms.

## Quality

- **Mach X kernels against stock operations on the same 8-bit weights:** 28.4 dB PSNR, 0.954 SSIM (49 frames at 768×512, where both
  are bit-exact run to run). Against the unquantized bf16 original: 26.0–26.2 dB / 0.899–0.903 for Mach X, 25.8 dB / 0.906 for the
  stock operations. The kernels add no loss of their own; the 8-bit weights themselves cost about 26 dB against bf16 (the same
  scenes, small changes in detail). Sound level and spectrum equal the stock run's at identical settings (RMS 0.303 against 0.293).
- A four-prompt gate (49 frames at 768×512, each against its own bf16 reference) gives PSNR 26.13 / 30.73 / 21.93 / 33.37 dB, mean
  28.04 dB, SSIM 0.884. A harmless change moved the mean by 0.84 dB, so differences under about 1 dB are noise on this gate.
- The VAE decode's per-channel RMS norm and SiLU as one kernel: 1.49 ms against 3.57 ms on a 264 MB tensor; error against float32
  9.76e-3 against stock's 9.92e-3.

## Known issues

- **About one clip in 30 comes out invalid** (all NaN; about 0.3 % of steps). Replaying the failing step's inputs six times gave
  finite output every time, so it is not data-dependent: a rare runtime fault. Ruled out: oneDNN's scratchpad for our multiply (0 bytes at
  every shape). Not yet separated: our two-card exchange, GPU compute, or the test machine's hardware. The script checks every clip
  (latents, pixels and sound finite and in range), refuses an invalid one, saves evidence, prints a warning and makes the clip again
  once with the same seed.
- **Long clips can differ between two identical runs.** The stock script does too: the variation is PyTorch XPU's own behaviour on these
  cards at some sizes (a bisect over every version, ours and stock, found it in all of them). Runs are bit-exact at 49 frames, 768×512.

## Measured and not adopted

- **Our own bf16 XMX flash attention** (head dim 128, seven variants, all correct to PyTorch's own error): 11.9–27.5 TFLOP/s
  against PyTorch's 91 (the best about 3.5× slower); not integrated. Ablating the exponent, the P·V product or the score matmul
  left the time unchanged.
- **Card-to-card copies on the GPUs** (event-ordered, no host wait): bit-exact, 5.4 GB/s a direction, no better than the host route.
- **A pure 8-bit integer multiply** (activations too): 2× on the multiply, but only about 2.6 ms a block gained net after the
  quantize and rescale passes, and lossy.
- **A float8 weight cast:** 1.0 ms a layer with 3.4 % mean error, against int8 with a per-row scale at 1.6 ms and 0.9 %.

## Not done

Two-stage 1080p, image-to-video, the non-distilled (dev) transformer, a quantized weights file to cut the 20 s load, and a fix for
the intermittent invalid clip.

Earlier on the same cards, Wan2.2-TI2V-5B took 456 s for 5 s at 720p without sound.
