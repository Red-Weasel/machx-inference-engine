# Model settings: what each model runs with, and what was measured there

One page per model the engine serves day to day on two Arc Pro B70s (32 GB each) with 256 GB of RAM: the file, the launch
that is used and tested, the sampler the engine fills in when a request omits a field (the publisher's card), the numbers
measured at that launch, and the switches worth knowing. Dates are when the figure was measured; the README's
"New in" sections carry the details. Launch columns are `ie serve` flags; Dream's model screen sets the same things.

Rules that hold for every model here:
- Two cards (`--gpus 2`) for every model below; the layer split is the default, tensor parallel is opt-in where it exists.
- `--vram-reserve-gib` stays at 0 (or the small value shown): a larger reserve trades decode for nothing on this machine.
- Model files live on an NVMe drive. A USB disk is the slowest device in the box; nothing serves from one.
- After installing a new engine build, restart the engine in Dream (it keeps the old process otherwise).
- Prompt reading is "prefill", writing is "decode"; both in tokens per second unless a time is given.

## Qwen3.8-Flash-Next (`qwen4exp`) — the daily driver with vision

| | |
|---|---|
| File | `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000N-of-00004.gguf` (104 GB), with `mtp-head.gguf` (the MTP draft head, 5.2 GB; made by `scripts/fetch_mtp_head.py` + `scripts/mtp_to_gguf.py`) and `mmproj-model-bf16.gguf` (the vision tower) beside it |
| Launch (tested) | `--gpus 2 --ctx 262000 --parallel 4 --slot-ctx 75000 --prefill-chunk 2048 --thinking on --reasoning-effort xhigh --spec --spec-k 3` (October 23); the same at `--parallel 1` for one conversation at a time |
| Sampler (card, thinking) | temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence 0, repeat 1.0; instruct: 0.7 / 0.80 / 20, presence 1.5 |
| Speculation | the MTP head: the default at one lane; on the lanes with `--spec` (Dream's speculation toggle). `--spec-k 3` is the certified draft length (Dream passes its `spec_k`). Exact at any temperature (speculative sampling); the tokens are plain decoding's |
| Decode, one conversation at the launch above | 9.8K-token conversation 28.3 / 27.2 / 26.5 tok/s over three turns (lanes awake on demand, no `--spec`), 29.5 / 28.0 / 27.2 with `--spec`; a 17.3K document 26.9 / 30.0; short prompts ~30 (v0.2.33, October 23). One lane with the head: greedy 35.6 → 38.9 short, 26.4 → 31.4 at 8–64K (v0.2.30) |
| Decode, several at once | three 256-token requests together 14.5 s (lazy lanes) / 16.9 s with `--spec`; four together 18.1 / 20.7 s; sixteen at 118.4 tok/s together with row batching (`--ctx 8192`: 128.0) |
| Prefill | 589 / 897 / 946 / 945 tok/s on unique 2K / 8K / 16K / 32K prompts at `--ctx 262000` (628 / 978 / 1,031 / 1,006 at `--ctx 40000`); a 9.8K conversation's first token in 13.0 s cold, 1.3–1.5 s from the prompt cache on later turns; a 17.3K document in 21 s |
| Vision | images on every request lane (v0.2.33); up to the 12,288-patch cap (about 2,700 × 2,700 px; larger images are refused with `IE_VIS_MAX_PX`'s message — downscale them); long image prompts read with the image since v0.2.21 |
| VRAM | expert cache per card: 224 / 227 resident experts a layer with one lane awake and no head bank, 191 / 162 with `--spec` (the head's Q8_0 bank takes ~35 slots of card 1, the vision tower ~17 of the card it sits on) |
| Switches | `IE_Q4E_MTP=0` no head; `IE_Q4E_LANES_MTP=0` head loaded, rounds off; `IE_Q4E_LANES_MTP_MAX=N` rounds while up to N lanes decode (default 1); `IE_Q4E_LAZY_LANES=0` the eager lane load; `IE_Q4E_PF_RING=R` the prefill expert ring (default 96; 0 = off); `IE_Q4E_PIPE_CHUNK=4096` faster long prompts but different logits (opt-in); `IE_Q4E_ROWS=0` one lane a card step; `IE_Q4E_SHARED_PREFIX_GIB` the shared-prefix store (default 2); `IE_MMPROJ` / `--mmproj` the tower file |
| Known | `--prefill-chunk` is ignored here: the pipelined prefill runs 2,048-token chunks; with three or four conversations decoding at once `--spec` costs ~17 % (the head's bank), so leave the toggle off for multi-agent bursts |

## Qwen3.8-35B-A3B Distill (`qwen35moe`) — the fast one

| | |
|---|---|
| File | `Qwen3.8-35B-A3B-Distill-abliterated-Q8_0.gguf` (36 GB); Q6_K and Q5_K native |
| Launch (tested) | `--gpus 2 --ctx 262144 --parallel 6 --slot-ctx 40000 --prefill-chunk 2048 --temp 0.6 --top-k 20 --top-p 0.95 --thinking on` (October 17 preset) |
| Sampler (card, thinking) | temperature 0.6 (the distill's best practice), top_p 0.95, top_k 20; instruct 0.7 / 0.80 / 20, presence 1.5 |
| Speculation | `--spec` = prompt-lookup drafts across the lanes (copies of the prompt); Dream offers it at temperature 0. The MTP head is not a lever here (the MoE verify rows re-read experts) |
| Decode | one request 90.6 tok/s; a 9.8K conversation 86.5 / 86.1 / 83.3 over three turns; a 17.3K document 82.0; three requests at once 71.7 each; sixteen at 414 tok/s together (v0.2.32, October 22; the lanes figure October 1) |
| Prefill | 2,896 / 2,808 tok/s at 2.4K / 10.7K; greedy bench at 2K / 9.8K / 17K / 32K prompts: 89.0 / 86.6 / 82.4 / 69.8 tok/s decode after them |
| Vision | with `--mmproj`, on every request lane since v0.2.27 |
| Switches | `IE_FA2_XMX_V2=0..3` the decode attention variants (3 = the default); `IE_Q35MOE_*` the lanes' scheduling knobs (FIFO, short-first, re-cut, decode quota, lookup); `--int8-kv` off (slower) |
| Known | the XML tool calls and reasoning come back in the OpenAI fields; one card runs it too (Q5_K_M 87.2 / 82.9 tok/s) |

## Qwen3.8-27B (`qwen35`, dense)

| | |
|---|---|
| Files | `Huihui-Qwen3.8-27B-abliterated-Q8_0.gguf` (28 GB), Q6_K (21 GB), Q5_K, a Q4_K_M re-quant (16 GB) |
| Launch (tested) | Q8_0: `--gpus 2 --ctx 222222 --parallel 8 --slot-ctx 60000 --prefill-chunk 2048 --thinking on --reasoning-effort xhigh --spec --spec-k 3` (October 18); Q6_K / Q5_K at `--ctx 250000 --parallel 8` / `16 --slot-ctx 50000` |
| Sampler (card, thinking) | temperature 1.0, top_p 0.95, top_k 20; instruct 0.7 / 0.80 / 20, presence 1.5 |
| Speculation | `--spec` = the MTP head, on the lanes too (v0.2.27), 3 rows a round by default; works with any sampling |
| Decode | Q8_0 at the launch above 28.7 tok/s (17.5 before the lanes allocated on demand, v0.2.30), later turns from the prompt cache in 0.6–0.75 s; Q4_K_M 27.5 plain, 37–40 with `--spec`, two together 63; Q6_K +28 % and Q5_K +39 % over Q8_0 plain (October 15) |
| Prefill | 2,962 / 3,001 tok/s at 2.4K / 10.7K (Q8_0); Q6_K 1,378 / 1,549 |
| Vision | with `--mmproj`, on every request lane |
| Switches | `IE_QWEN35_LANES=0` the one-at-a-time path; `IE_QWEN35_LANES_MTP=0` rounds off, `IE_QWEN35_LANES_MTP_MAX` (default: the card count); `IE_QWEN35_LANES_EAGER=1` the eager lane load |
| Known | `--int8-kv` is slower on this dense model; tensor parallel gives nothing here |

## Nanbeige4.3-33B (`llama`, dense) -- ported October 28, abliterated and given the dense path's serving features October 29

| | |
|---|---|
| Files | `~/models/Nanbeige4.3-33B-abliterated-GGUF/Nanbeige4.3-33B-abliterated-Q6_K.gguf` (27 GB). OUR build: the authors' BF16 release (`Nanbeige/Nanbeige4.3-33B`, 14 shards checksummed against the Hub's records) with a per-layer refusal-direction edit of 20 tensors (layers 18-27's o_proj and down_proj, the direction read at the `assistant` role token), converted and quantized here (`~/models/abliterate-nanbeige43-33b/`, LOG.md). Held-out report: harmful refusals 46/48 -> 7/48 thinking off, harmless 0/24 -> 0/24; on the engine with thinking on (the daily mode) 9 of 24 still refuse -- a stronger window is the next iteration. Q6_K is the native quant on the dense path (Q8_0 / Q5_K would be expanded to F16) |
| Launch (tested) | `--gpus 2 --ctx 65536 --thinking on` (October 29): 28-29 tok/s decode, the prompt cache on by default, `--spec` for the lookup drafts. One card: `--ctx 8192` is the most that fits (27 GB of weights + the lm_head repack), 19.3 tok/s |
| Sampler | none on the card: the engine fills the library defaults. Keep thinking on (the default); with it off the model emits a stray `</think>` and repeats its sentence -- llama.cpp does the same on the identical render |
| Decode | two cards 28.7 / 27.8 tok/s at 901 / 5261 tokens of context (24.2 / 23.6 on October 28; 6.3 / 2.6 before the port's pass): the decode body now runs DenseModel's launch sequence (the fused norm + Q8 quant, the SoA int-dot lm_head). One card 19.3 / 18.5 at ctx 8192 |
| Prefill | two cards 367 / 254 tok/s at 901 / 5261; one card 224 / 150 (the server's numbers; `ie-bench`'s 307 is DenseModel's own path, known_bugs 39) |
| Prompt cache | on by default (FleetPrefixCache on the fleet class, one card or two): turn 2 of a 5.4K-token conversation restores 5250 of 5414 tokens in 6 ms (the insert 60 ms), prefill 21.4 s -> 1.7 s, the warm reply byte-identical to a cold one. `IE_NO_PROMPT_CACHE=1` off; `IE_PROMPT_CACHE_MAX_ENTRIES` (12), `IE_PROMPT_CACHE_MAX_PREFIX`, `IE_PROMPT_CACHE_VRAM_MIB` |
| `--spec` | prompt-lookup drafts at --parallel 1 (rounds of up to 16 rows through the batched SoA lm_head): a repeated text decodes at 77.6 tok/s on two cards and 49.9 on one (2.7x / 2.6x plain), 515 of 539 drafts accepted, 7 tokens a forward; prompts without repetition decode at exactly the plain speed; the texts are byte-identical to plain on every gate request. `IE_DENSE_LOOKUP=0` off, `IE_DENSE_LOOKUP_MIN` (16), `IE_DENSE_LOOKUP_ROWS` (16; 8..16), `IE_DENSE_LOOKUP_STATS=1` a line per request |
| Context | fp16 KV 212 KB a token (52 layers x 8 kv heads x 128 x 2); `--gpus 2 --ctx 65536` ~13.6 GB of weights + ~6.9 GB of KV a card, ~9 GB a card left for the prompt cache |
| Switches | `IE_DENSE_ONE_CARD_FLEET=0` the previous one-card path (DenseModel: no cache, no `--spec`; it is also the path of `--int8-kv` and of Q8_0 / Q5_K files); `IE_DENSE_KQ_C32=0` the old Q6_K kernels |
| Known | no request lanes yet (`--parallel N` queues; P3 of the serving campaign); with the cache on the prompt's last few tokens run as their own small forward (the int-dot rows kernel, naive attention), so one-card greedy text drifts at a near-tie after ~50-80 words against a cache-less prefill and a warm turn's wall is that tail (~95 tok/s for 164 tokens) -- known_bugs 38; `--int8-kv` keeps the fp16 cache too (no VRAM saving); thinking-on refusals on the held-out set ~37 % (thinking off 15 %) |

## DeepSeek-V4.1-Flash (`deepseek41`)

| | |
|---|---|
| File | `DeepSeek-V4.1-Flash-UNCENSORED` safetensors (475 GB, FP8 dense + MXFP4 experts; the October 17 weights swapped in place), host-resident experts in 256 GB of RAM |
| Launch (tested) | `--gpus 2 --ctx 265000 --parallel 4 --slot-ctx 75000 --prefill-chunk 2048 --thinking on --reasoning-effort high --vram-reserve-gib 0` (October 20) |
| Sampler (card) | temperature 1.0, top_p 0.95, top_k 0 (off), min_p 0; effort `max` for the card's agent and coding benchmark setting |
| Speculation | prompt-lookup on by default (`IE_DS41_LOOKUP=0` off) |
| Decode | 12.7–15.3 tok/s for one request (it varies 8–15 by prompt; the first request after a load is cold, discard it); two requests 25.1–25.8 together; the elastic VRAM region on by default (+13 %) |
| Prefill | 304 tok/s at 2K, 319 at 32K–223K (September); tensor parallel (`IE_DS41_TP=1`, opt-in): decode +18 %, long-prompt reading −41 % |
| Vision | native, at `--parallel 1` |
| Switches | `IE_DS41_TP=1` tensor parallel; `IE_DS41_ELASTIC_GIB=0` the elastic region off; `IE_DS41_CONT` continuation chunks (default on); `IE_DS41_CPU_MISS=0` for an A/B at the token level (the default mode's CPU leg drifts) |
| Known | never raise `IE_DS41_MAX_FWD` under tensor parallel at long context (it took the desktop down once); the reserve stays 0 |

## Xiaomi MiMo-V2.6-Flash-RL (`mimo26`)

| | |
|---|---|
| File | `MiMo-V2.6-Flash-RL` safetensors (178 GB, FP8 dense + MXFP4 experts); an UNCENSORED copy hard-linked beside it |
| Launch (tested) | `--gpus 2 --ctx 250000 --parallel 1 --slot-ctx 75000 --prefill-chunk 2048 --thinking on --vram-reserve-gib 1` (October 14; the base copy at `--ctx 200000 --vram-reserve-gib 0.5`) |
| Sampler (card) | temperature 1.0, top_p 0.95; the other fields off |
| Speculation | its bundled DFlash drafter, on by default |
| Decode | 27.8–28.3 tok/s with the drafter on agent-style prompts, 16.9–18.9 without (v0.2.23); four requests at once 25.6 together |
| Prefill | 596–630 tok/s at 5–6K, 708–716 at 38K (`ie serve`) |
| Vision | native, with the tower |
| Known | greedy output is not deterministic run to run (the CPU expert-miss leg); very sensitive to small numeric changes on agent transcripts; the reserve stays small |

## GLM-5.3-Flash (`glm5next`)

| | |
|---|---|
| File | `GLM-5.3-Flash-UD-Q4_K_XL-0000N-of-00006.gguf`, host-resident experts |
| Launch (tested) | `--gpus 2 --ctx 264000 --parallel 1 --prefill-chunk 1024 --thinking on --reasoning-effort max` (October 4) |
| Sampler (card) | temperature 1.0, top_p 0.95, top_k 0; GLM always reasons (pick the effort) |
| Speculation | the MTP draft on card 0 beside card 1's step, on by default (`IE_G5_PIPEDRAFT=0` off) |
| Decode | 13.5–14.3 tok/s (v0.2.26, October 20; 10.8–11.5 before) |
| Prefill | a 9.6K agent prompt at 300 tok/s (first token 32.4 s, was 243 s); later turns restore from the prompt cache in 9.5–9.7 s |
| Known | `ie serve` runs GLM one request at a time; the CPU expert leg (`IE_G5_CPU_MISS=0` for a bit-exact A/B) |

## DeepSeek-V4-Flash (`deepseek4`)

| | |
|---|---|
| Files | `DeepSeek-V4-Flash-0731-MXFP4` GGUF (155 GB); the Vision-Exp abliterated MXFP4 variant |
| Launch (tested) | `--gpus 2 --ctx 300000 --parallel 1 --prefill-chunk 2048 --thinking on --reasoning-effort high` (the vision variant `--prefill-chunk 512`) |
| Sampler (card) | temperature 1.0, top_p 0.95, top_k 0 |
| Decode / prefill | 26.1 tok/s at 4K, 32.7 on a short prompt; prefill 571 tok/s at 4K (September 11); the vision variant: 28 decode, 548 prefill (September 2) |
| Known | images at `--parallel 1` only; measured on an earlier driver stack |

## Also running

| Model | Arch | Launch / state | Measured |
|---|---|---|---|
| Laguna-S-2.1 | `laguna` | two cards (October 18); sampler temperature 1.0, top_p 1.0, top_k 20 | `ie serve` 35 tok/s; decode 20.8 → 32.9 one request |
| Inkling | `inkling` | two cards (October 18) | serve 23–28 tok/s; prefill +45 % |
| Holo4 27B / 35B (computer-use) | `qwen35` / `qwen35moe` | the 27B / 35B launches above | runs; the 35B verified October 15 |
| gpt-oss 120B / 20B, Gemma-4 31B / 26B-A4B, Qwen3 / Qwen3-Next / Qwen2.5, Llama / Mistral / Phi / Granite GGUFs | various | generic GGUF path | see the README's supported architectures |

Sources: the README (v0.2.21 – v0.2.33), the engine's `recommended_sampling.hpp`, and the launches saved in Dream's model presets (October 4–23, 2026). Figures marked by date are single measured runs at that launch, not averages.
