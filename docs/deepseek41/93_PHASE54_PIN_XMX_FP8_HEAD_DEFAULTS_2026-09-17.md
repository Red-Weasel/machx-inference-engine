# V4.1 — Phase 54: three founder decisions made default (+10 GiB pinned, XMX prefill experts, an FP8 LM head)

Founder, 2026-09-17 02:02: "approve the extra 10 GiB pinned RAM and make it default, assuming theres space. then
implement 2 and 3" (2 = the XMX prefill expert route, 3 = a lossy FP8 LM head; docs/92).

## 1. Pinned expert arena: +10 GiB, with the free floor it was measured at

`Ds41Forward::init_resident`: the shared live rule's cap (total - 66.84 GB, or MemAvailable - 33.42 GB) plus 10 GiB,
then min with **MemAvailable - 30 GiB** (was 40): a box with less free memory at load gets a smaller pin, not a squeeze.
`IE_DS41_PIN_EXTRA_GIB` overrides (0 = the shared rule). At ctx 75,000: **280 pinned experts per layer (+14)**, cap
196.8 GiB, MemAvailable floor during serving 29-30 GiB.

## 2. XMX prefill expert route: default on

`IE_DS41_PREFILL_XMX=0` restores int-dot W4A8. The decode and cont tests, whose near-tie / chunk-vs-step bars were
derived on int-dot (docs/82: 26/28 and 14/15 on XMX by design), pin the int-dot route unless the caller names one.

## 3. FP8 LM head

`Ds41DenseCache::upload_head(..., fp8_head)`: the BF16 head [129,280 x 5,120] quantised at load (16 threads, ~1 s) to
E4M3 codes (round to nearest; 0x7F NaN excluded) with one E8M0 exponent per 32x32 block = the smallest power of two
keeping the block's max magnitude within 448 -- the checkpoint's own dense FP8 layout, so the tuned FP8 GEMVs read it:
the relay GEMV for one row (decode, last-row prefill), the rows kernel for 2-8 decode rows, the cache's fp16 dequant
scratch for a full-logits prefill. 662 MB instead of 1.32 GB: card 1 gets **2 more static slots per layer (51 -> 53)**.
On in `ie serve` and `ie-ds41-run` (`ResidentOptions::head_fp8`); `IE_DS41_HEAD_FP8=0/1` overrides; the reference
tests keep BF16.

**Quality (paired, wikitext-2 test, 16,384 tokens, exact path, both on the XMX route):** BF16 head PPL 1.88776, FP8
head **1.88640**: -0.000717 nats/token, block SE 0.000302 (t = -2.37) -- no loss measurable (the sign is rounding
luck, not a gain); top-1 agreement 99.46 %.

## The expert file follows the new boundary

Both cards now start the disk tier at rank 333 (53 static + 280 pinned). File rebuilt at [333, 377): covers 880 of
1,020 mmap experts per card (86 %).

## Decode, held-out chat set (6 prompts, `ie serve`, fresh process per arm)

| arm | tok/s | MemAvailable floor |
|---|---|---|
| new defaults, run 1 | **12.76** | 29 GiB |
| same, BF16 head | 12.32 | 30 GiB |
| new defaults, run 2 | **12.84** | 30 GiB |
| (docs/90/92) old defaults | 12.22 / 12.52 / 12.63 | ~38 GiB |
| (docs/92) +10 GiB alone, BF16 head, int-dot prefill | 13.08 | 30 GiB |

The FP8 head is **+4 %** in a same-configuration A-B-A. Across configurations that change the PREFILL route the
comparison is confounded: the cached prompt state is computed by different arithmetic, the generated text drifts, and
decode speed moves with the text by about +-5 % -- so the three changes together read as ~+3 % over the old defaults
here, below the parts measured alone. First-turn prefill of the 17.7k-token prompt: 281 tok/s (was 270).

## Gates (build of 02:25; `ds41_work/p54/`)

| gate | result |
|---|---|
| decode (int-dot pinned) | **28/28** |
| decode on the XMX route | 26/28 -- the two documented misses (docs/82) |
| decode with the FP8 head | 28/28 (informational: the head is within the logits bars) |
| multi / rollback / dspark | **38/38, 74/74, 153/153** |
| forward (the transformers reference, XMX default) | **5/5** |
| cont (int-dot pinned) / cont on XMX | **15/15** / 15/15 (Phase 42's 14/15 miss no longer reproduces) |
| generate on the default (XMX) route | 14/15: T = 512 prefill vs prefill(T-2)+2 steps, KL 0.0031 < 0.02 but a near-tie argmax flip |
| generate on int-dot | **15/15** (KL 0.0049, same argmax) -- the test now pins int-dot like decode and cont |
| cache (prefix cache bit-identity, XMX default) | **21/21** |
| resident | **14/14**; pp2048 warm **304.1 tok/s** (290.8 before), cold 249.8 |
