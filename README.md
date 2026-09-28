
# Mach X — LLM Inference Engine for Intel Arc

**A C++/SYCL local LLM inference engine for Intel Arc GPUs, built and tuned on two Arc Pro B70 cards. XMX kernels, quantized models and multi-GPU execution. New in v0.2.0: several models behind one endpoint, and several requests decoding at once.**

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Language](https://img.shields.io/badge/C%2B%2B20-SYCL%20%2F%20DPC%2B%2B-orange)
![Platform](https://img.shields.io/badge/Intel%20Arc-Battlemage%20B70-0071C5?logo=intel&logoColor=white)
[![Model coverage](https://img.shields.io/badge/models-dense%20%2B%20MoE%20%2B%20hybrid-brightgreen)](#supported-architectures)
![Vision](https://img.shields.io/badge/vision-VLM%20ready-purple)
![Multi-GPU](https://img.shields.io/badge/multi--GPU-tensor--parallel-success)
[![Release](https://img.shields.io/github/v/release/Red-Weasel/machx-inference-engine)](https://github.com/Red-Weasel/machx-inference-engine/releases)

Intel Arc is a genuinely capable AI GPU that inference tooling has mostly ignored. **Mach X is built for it from the metal up** — no fork of llama.cpp, no PyTorch, no vendor runtime. Hand-written SYCL kernels (XMX matrix engines, int-dot quantized GEMV, tiled FlashAttention), an OpenAI-compatible server, tensor-parallel multi-GPU, and day-one support for the newest model architectures — often running them fast on Arc *before* anyone else does.

---

## ⚡ What people run on it

The six models this engine is tuned for, each on **two Arc Pro B70 cards** (64 GB VRAM) with host RAM holding the
experts that do not fit. Dates, workloads and methods are in [Benchmarks](#benchmarks).
Figures without a date were measured on the driver stack v0.2.0 ships on (September 26–27, 2026); dated ones are from
the previous driver stack.

| model | weights | prefill | decode | also |
|---|---|---:|---:|---|
| **MiMo-V2.6-Flash** | 178 GB safetensors (FP8 dense, MXFP4 experts), 256 GB RAM | **438–483** tok/s at 4–6K; **434–438** at 32K, **322** at 120K (September 22) | **23.1–23.9** tok/s chat, **16.7–16.9** without the drafter; **24.3** on agent-style copy edits (September 22); with `ie serve --parallel 2` / `4`, **28.38** / **28.46** tok/s together (×1.25 / ×1.27 against one drafted request in the same run; September 26) | released and running the same day: its bundled DFlash drafter (speculative decoding), **native vision**, tool calls, thinking on/off, 120K context verified, other conversations kept in host memory, up to 16 requests at once in `ie serve` (`--parallel N`, [below](#several-requests-at-once-request-lanes)). |
| **DeepSeek-V4.1-Flash** | 475 GB safetensors (FP8 dense, MXFP4 experts), 256 GB RAM | **304** tok/s at 2K, **319** at 32K–223K (September 16–17) | **12.8** tok/s chat, **14.0** in agent loops (September 17–18); with `ie serve --parallel 2`, **25.1–25.8** tok/s together (×1.68–1.69 against one request in the same run) | native vision, tool calls, 223K context, 1–2 s follow-up turns from the prompt cache, a disk prompt cache that survives rebuilds, up to 16 requests at once in `ie serve` (`--parallel N`, [below](#several-requests-at-once-request-lanes); 25.7 tok/s together at 16). |
| **DeepSeek-V4-Flash** | 155 GB GGUF (MXFP4 experts, Q8_0 dense) | **571** tok/s at 4K | **26.1** tok/s at 4K, **32.7** short | tool calls, prompt cache; measured on the previous driver stack (September 11, 2026) |
| **GLM-5.3-Flash** | UD-Q4_K_XL GGUF, host-resident experts | **156.4** tok/s at 16K | **14.0–14.4** tok/s at 16K | MTP draft, two-GPU pipelined prefill, request lanes in a test tool (two lanes ×1.68 against one request; `ie serve` runs GLM one request at a time) |
| **Qwen3.8-Flash** (Flash-Next) | 104 GB UD-Q4_K_XL GGUF | **495–502** tok/s pipelined, once warm | **37.1–38.2** tok/s chat, **44.4–44.6** code (lossless speculative) | native vision, up to 16 requests at once in `ie serve` (two at **59.0–62.6** tok/s together, ×1.94–2.06; 16 at **118.4** with row batching) |
| **Qwen3.8-27B** | Q8_0 GGUF | **945** tok/s at 2K | **24.5** tok/s (tensor-parallel + speculative); both August 15–26 | prompt cache (layer-split), up to 16 requests at once in `ie serve` with row batching (16 at **108.0** tok/s together, 17.1 alone) |

Everything runs behind one OpenAI-compatible server (`ie serve`) with tool calls. New in v0.2.0, `ie supervise` puts
several servers — one per card set — behind **one endpoint that routes by model name**, and the
[Dream Agent Harness](https://github.com/Red-Weasel/Dream-Agent-Harness) drives either one as a local agent.

![DeepSeek-V4.1-Flash running locally in the Dream Agent Harness, served by Mach X on two Arc Pro B70 cards](docs/images/dream-deepseek-v41.png)
<sub>DeepSeek-V4.1-Flash on two Arc Pro B70 cards, served by `ie serve` and driven from Dream — reasoning shown, 11.7 tok/s.</sub>

---

<a id="new-in-v020"></a>
## 🆕 New in v0.2.0

Everything since [v0.1.0](https://github.com/Red-Weasel/machx-inference-engine/releases/tag/v0.1.0) (September 21, 2026).
The [release notes](https://github.com/Red-Weasel/machx-inference-engine/releases/tag/v0.2.0) list every change,
including the few behaviour changes. The layout, card and supervisor features are additive: without `--config` and
`--cards`, the engine parses its arguments and picks its GPUs exactly as before.

### Several models on one endpoint

- **`ie supervise --config <layout.json>`** starts every server of a layout as its own `ie serve` child, each with its
  own model, cards, port, context and prompt cache, and serves **one OpenAI-compatible endpoint** in front of them. A
  request goes to the server its `"model"` names (or the layout's `default`); streams pass through byte for byte, and a
  client that disconnects is disconnected upstream. `GET /v1/models` lists every server and whether it is loading or
  ready, `GET /health` aggregates them, and `GET /props?model=<name>` is that server's own `/props`. A server that
  crashes is reported while the others keep serving; `"restart": "on-failure"` restarts it (at most 3 times, never after
  a lost device). Shutdown asks every child to stop and never kills one: killing a process that holds a model can
  wedge the GPU.
- **Layout files for `ie serve --config`.** A JSON file holds a server's model, cards, port, context and every launch
  flag, plus environment variables; it is validated by the same parser as the command line, and flags on the command
  line win.
- **`--cards` pins a process to chosen GPUs**, and **`ie cards`** lists them without loading anything. At most one
  engine process holds a card at a time (a per-card lock). Pinned to card 1, a server put 2,292–2,363 MiB on card 1
  while card 0 stayed at 28 MiB; two servers, one per card, loaded side by side and answered concurrent requests.
- **`ie --help`, `ie help <command>` and `ie <command> --help`** print every flag, its default, the architectures it
  applies to and the environment variables that matter, without loading a model.

A card belongs to one server at a time, so on a two-card machine a model served across both cards (as DeepSeek-V4.1
and MiMo are) is the whole layout, and smaller models can run one per card. [Layouts, routing, card locks and
limits](docs/serve_config.md).

<a id="several-requests-at-once-request-lanes"></a>
### Several requests at once: request lanes

`ie serve --parallel N` serves up to **16** requests at once on MiMo-V2.6-Flash, DeepSeek-V4.1-Flash, Qwen3.8-Flash,
the Qwen3.6-35B-A3B class and Qwen3.8-27B.
Each model checks at load whether N lanes fit its memory and refuses, with the numbers, a count that does not.
New for the 35B-A3B class, Qwen3.8-Flash and Qwen3.8-27B is **row batching**: the decoding lanes' one-token steps are
grouped, and a group runs as one card step that reads the weights once. Every reply is byte-identical to the same
request run alone. Here and below, "byte-identical" and "the previous server byte for byte" compare requests with the
same sampling settings ([recommended sampling](#recommended-sampling-per-model)).

| model | 1 request | 4 at once | 8 at once | 16 at once | |
|---|---:|---:|---:|---:|---|
| Qwen3.6-35B-A3B class | 60.8 tok/s | 178.8 | 249.2 | **310.2** | ×5.11 at 16 |
| Qwen3.8-Flash | 28.0 | 74.7 | 97.7 | **118.4** | ×4.24 at 16; long counting-task replies |
| Qwen3.8-27B | 17.1 | 60.0 | 88.4 | **108.0** | the previous joint-step path: 49.3 / 60.0 / 63.4 at 4 / 8 / 16 |
| MiMo-V2.6-Flash | ~20 | 22.8 | – | 24.7 | flat (DFlash drafter on, the default) |
| DeepSeek-V4.1-Flash | ~13 | 22.6 | – | 25.7 | ×1.92 at 16 |

Tokens per second summed over the requests, on the release-candidate build, new driver stack, September 27–28, 2026:
`ie serve --ctx 32768 --parallel 16` with lanes of 32,768 positions (the 27B `--ctx 16384 --slot-ctx 4096`), served
defaults, greedy, short prompts and long replies, one A-B-A run per N (one request alone, N together, one alone).
Qwen3.8-Flash's figures depend on the workload (through its expert-cache hits). DeepSeek-V4.1 got a long tutorial
task instead of the counting task, which it declines.
With row batching off, Qwen3.8-Flash makes 60.9 tok/s at 4 and 67.1 at 16.

What each agent sees: tokens per second per request, approximate, assuming an even share (the total above ÷ N):

| model | N=4, each | N=8, each | N=16, each | alone |
|---|---:|---:|---:|---:|
| Qwen3.6-35B-A3B class | 44.7 | 31.2 | 19.4 | 60.8 |
| Qwen3.8-Flash | 18.7 | 12.2 | 7.4 | 28.0 |
| Qwen3.8-27B | 15.0 | 11.1 | 6.75 | 17.1 |

Choosing N: use few lanes when each agent needs to be fast, and many lanes for bulk throughput.

**Why MiMo and DeepSeek-V4.1 scale less.** Their experts live partly in host RAM: a token's experts that are not on
the card come over PCIe or run on the CPU expert leg, and with two or more requests that host side, not the cards,
sets the pace. Lanes share out that fixed budget of expert work instead of adding to it; on V4.1 the CPU leg is
estimated at about 94 % busy at 16 lanes. On MiMo the drafter already fills a step's 8 rows with a single request, so
with drafting on the aggregate stays flat (with the drafter off, an earlier run reached 29.1 tok/s at 16).

**Shared-prefix cache.** On the 35B-A3B class and Qwen3.8-Flash (on by default), and on Qwen3.8-27B (opt-in), a system
prompt plus tool schemas that several conversations share (1,024 tokens or more) is cached once and restored into any lane. Four agents arriving at
once with a 13.1K-token shared prefix (16 tools) got their first tokens in 2.9–3.1 s instead of 60.5 s on the 35B-A3B
class, and with an 8.7K-token prefix on Qwen3.8-Flash in 4.2–4.5 s instead of 65.6 s. The output is byte-identical to
a cold prefill. `IE_SHARED_PREFIX=0` turns it off. On the 27B, `IE_QWEN35_SHARED_PREFIX=1` turns it on; it is off by
default there because it splits the 27B's prefill at the shared prefix, which changes that model's single-request output
for long system prompts.

**Context per lane.** 16 lanes of 64K context fit on the 35B-A3B class (31.06 GB used per card) and on Qwen3.8-Flash.
On Qwen3.8-Flash every lane's context comes out of the expert cache, so 64K lanes decode about 10 % slower than 8K lanes
(20.7 against 23.0 tok/s alone, 47.0 against 51.4 at 16, measured before row batching); on the 35B-A3B class the lane
size does not change short-prompt speed.

**Switches.** `IE_Q35MOE_ROWS=0`, `IE_Q4E_ROWS=0` and `IE_QWEN35_ROWS=0` turn row batching off (one step per lane);
`IE_QWEN35_LANES=0` puts the 27B back on its joint-step path; `IE_SHARED_PREFIX=0` as above. Opt-in:
`IE_QWEN35_SHARED_PREFIX=1` (above); `IE_LANES_PREFILL_FIFO=1` (with `IE_LANES_HANDOVER_MS`) hands the cards back to the decoding lanes between new prompts'
prefills: in a burst of four ~13K-token prompts beside four decoding lanes, the longest decode pause fell from 27.8 s to
11.4 s, while the burst's first tokens came at 45–91 s instead of 52 s. `IE_MIMO26_DFLASH_MAX_LANES=5` stops MiMo's
drafting while five or more lanes decode (off by default: it was slower at 8 lanes).

**MiMo lanes and short prefixes.** At `--parallel` > 1, MiMo no longer reuses a cached prefix shorter than 1,024 tokens,
so a reply no longer depends on which lane served it or in what order requests arrived; short multi-turn chats re-read
up to 1,023 tokens a turn. `--parallel 1` is unchanged.

**Limits.** A long new prompt's prefill pauses the decoding lanes until it ends (see the opt-in above). With the drafter
on (the default), MiMo can give a different reply beside other requests than alone on some sampled or thinking prompts:
1 of 16 requests at 16 lanes, and 2 of the release gate's prompts at 4 lanes; with the drafter off
(`IE_MIMO26_DFLASH=0`) every check passes. On DeepSeek-V4.1 at `--parallel` > 1 a follow-up turn reuses a lane's cached
prefix and can differ from the same turn read cold at `--parallel 1` (pre-existing: the previous build gives the same
bytes; batch equals solo within one server). MiMo and DeepSeek-V4.1 are bound by the CPU expert leg, not the cards.

The first lanes were MiMo's; the per-model notes below date from September 26–27 and, unless the table above repeats
them, were measured with two to four lanes.

A single MiMo-V2.6-Flash request keeps each card busy only 51–64 % of the time.
v0.2.0 gives the MiMo forward **lanes**: each lane has its own attention caches, sliding-window rings and drafter
context, carved out of the static expert tier; one extra lane at 32K context costs about 1.34 GB across both cards.
With two lanes, card 0 runs one lane's step while card 1 runs the other's, and each card is 87–99 % busy.
With four lanes, two lanes' rows go through one forward together; every shared operation is bit-identical row by row
to a one-row call.

| lanes | aggregate decode | against one request |
|---:|---:|---:|
| 1 | 15.1–17.0 tok/s | — |
| 2 | 28.1–28.2 tok/s | **×1.66–1.69** |
| 4 | 28.9–30.0 tok/s | **×1.85–1.90** |

Plain decoding with the CPU expert path on, as served; A-B-A runs of the lanes test tool on held-out Dream prompts.
With the CPU expert path off, every lane's output is byte-identical to the same request run alone,
and with one lane (the default) the engine is the previous one: the same tokens and VRAM use, at the same speed within
run-to-run noise.
MiMo's single-request default also runs the DFlash drafter (23.6 tok/s), so the served gain over one drafted request is
smaller than these ratios: served, with the lanes drafting too, two requests together make ×1.25 of one drafted request
and four ×1.27 (below).

- **`ie serve --parallel N` for MiMo-V2.6-Flash** (N up to 16). The default,
  `--parallel 1`, is the previous server byte for byte, streamed responses included (ids and timings aside).
  Lane 0 keeps `--ctx` and every other lane gets `--slot-ctx` positions (default 32,768); a load whose lanes do not fit
  is refused with the numbers.
  The lanes draft with DFlash within a budget of 8 rows per step: a request decoding alone drafts up to 7 tokens as
  before, two decoding requests up to 7 each, three or four up to 3 each (`IE_MIMO26_DRAFT_BUDGET=0` drafts only for a
  request decoding alone).
  On held-out Dream transcripts two requests decode at **28.38 tok/s** together, 14.2 tok/s each, and four at
  **28.46 tok/s**, 7.1 each: ×1.25 and ×1.27 against one drafted request on the same prompts.
  A 1.6K-token prompt arriving while three requests decode got its first token in 4.4 s, within one prefill chunk of its
  time alone, and the decoding requests pause for about that long (measured before the draft budget).
  With the CPU expert path and the drafter off, every request's reply equals its solo run and a cancelled request leaves
  the others unchanged; over 20 requests host memory grew 17 MiB after the first wave, and VRAM and threads stayed flat.
  Limits: images are served at `--parallel 1` only; prompt lookup is off above 1; with the drafter on, a greedy reply can
  change at a near-tie when other requests decode beside it; and a request prefers a lane with room for its reply, but
  when only a smaller lane is free its reply can be cut at that lane's size.
- **`ie serve --parallel N` for DeepSeek-V4.1-Flash** (N up to 16, on both cards). The default, `--parallel 1`, is the
  previous server byte for byte, streamed responses included (ids and timings aside). Lane 0 keeps `--ctx` and every
  other lane gets `--slot-ctx` positions (default 32,768), out of the static expert tier. Prompt lookup works per request,
  as with one lane.
  Two requests decode at **24.92 tok/s** together, 12.46 each, against 14.81 / 13.99 tok/s for one request (×1.73); more
  lanes add room for more requests, and little speed (22.6 tok/s at 4, 25.7 at 16, table above).
  A 1.7K-token prompt arriving while two requests decode got its first token in 7.9 s, against 7.1 s alone; over 20
  requests host memory grew 10 MiB after the first wave, VRAM 10 MiB per card, and threads stayed at 119.
  With the CPU expert path off, every request's reply equals its solo run, prompt lookup on or off, and a cancelled
  request leaves the others unchanged.
  Images work with several lanes: a prompt with images prefills in its own turn while the other requests wait.
  Limits: with more than one lane the load refuses DSpark (`IE_DS41_SPEC=1`), `IE_DS41_PROFILE_OUT`,
  `IE_DS41_DUMP_ROUTING`, expert parallel (`IE_DS41_EP`) and a single card; the lanes were measured with `--ctx 32768` and
  prompts up to 8K tokens, not at the 75,000-token context of the quick start below. The same lanes also run in the
  `ie-ds41-lanes-test` tool (Benchmarks).
- **GLM-5.3-Flash:** request lanes and a two-card pipe, in a test tool (`ie-glm-lanes-test`), off by default. Two lanes
  decode at 22.26 tok/s together against 13.28 for one request (×1.676, in the configuration `ie serve` uses); with the
  CPU expert path off, each lane's output is byte-identical to its solo run. `ie serve` runs GLM one request at a time
  in this release. [Measurements](docs/glm53/P4_B7_LANES.md).
- **`ie serve --parallel N` for Qwen3.8-Flash** (N up to 16, on both cards), the first model on a **shared request-lanes
  module** (admission, lane choice, the card pipe, cancel and shutdown in one place) that the 35B-A3B class and the 27B
  below also use.
  Each lane holds its own attention, indexer and DeltaNet state, taken out of the expert cache (about 0.55 GiB per card
  per extra lane at 32K); `--parallel 1`, the default, is the previous server byte for byte, and at `--parallel 2` every
  request, sampled ones included, streamed the same bytes as at `--parallel 1`.
  Two requests decode at **59.0–62.6 tok/s** together against 29.0–31.9 for one (×1.94–2.06, two runs, the base model,
  short prompts); with row batching, 16 decode at 118.4 tok/s together (table above). Over 16 requests after the first
  four, host memory grew 1 MiB, and VRAM and threads stayed flat.
  Limits: images are served at `--parallel 1` only; while a long prompt prefills beside them, the other requests slow sharply or
  pause. `IE_Q4E_LANES=0` restores the old time-sliced path, `IE_Q4E_ROWS=0` one step per lane.
- **`ie serve --parallel N` for Qwen3.6-35B-A3B-class models** (`qwen35moe`, split over both cards, N up to 16) on the
  same module.
  An extra lane at 32K takes 0.34 GiB per card; `--parallel 1` is the previous server byte for byte, and at `--parallel 2`
  sampled, tool-call, thinking, follow-up-turn and 8K-token requests all equal `--parallel 1`. Before row batching two
  requests decoded at **117.09 tok/s** together against about 61 for one (**×1.91**), three at 117.24 (×1.92); with it,
  4 decode at 178.8 and 16 at 310.2 (table above). A card holds about 2.8 GiB
  more VRAM at `--parallel 2` before any request (the lane plus a prefill workspace sized up front). Measured on a Q8_0
  fine-tune of Qwen3.6-35B-A3B; this split has no vision path. `IE_Q35MOE_LANES=0` restores the one-at-a-time path,
  `IE_Q35MOE_ROWS=0` one step per lane.
- **Qwen3.8-27B** now serves `--parallel N` (up to 16) on the shared lanes module with row batching: 16 requests decode
  at 108.0 tok/s together against 17.1 alone, where its previous joint-step path reached 63.4 (table above), every reply
  byte-identical to the same request alone. `IE_QWEN35_LANES=0` restores the joint-step path; on it, two requests
  decoded at 32.28 tok/s together against 17.32 for one (×1.86, September 26), and a long prompt held up the other
  request while it prefilled (13.0 tok/s decode during a 3.4K-token prefill).
- **Images at `--parallel` > 1:** where the lanes cannot take images (MiMo-V2.6-Flash, Qwen3.8-Flash, and
  DeepSeek-V4-Flash with its vision sidecar), `/props` says vision is not ready and an image request gets HTTP 400
  `vision_not_ready` before admission. The DeepSeek-V4-Flash case was verified with host tests and a code probe, not
  with the model loaded.

[Lanes](docs/mimo26/P4_B1_LANES.md), [the card pipeline](docs/mimo26/P4_B2_PIPELINE.md),
[row groups](docs/mimo26/P4_B3_ROWS.md), [serving with `--parallel N`](docs/mimo26/P4_B4_SERVE.md),
[the draft budget](docs/mimo26/P4_B5_DRAFT_BUDGET.md), [DeepSeek-V4.1 serving](docs/deepseek41/P4_B6B_SERVE.md).

<a id="recommended-sampling-per-model"></a>
### Recommended sampling per model

- **Requests that omit sampling fields now use each model's published recommendation for thinking or instruct mode**,
  so sampled replies differ from v0.1; pass the fields, or set `--temp/--top-k/...` or `IE_SERVE_*`, to keep the old
  0.7/40/0.95. The order is the request's value, then a command-line flag or `IE_SERVE_*`, then the model card's value
  for the request's mode, then the library default. Greedy requests (a temperature of 0) and requests that set
  `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty` and the repeat penalty are unchanged; `max_tokens` and
  `frequency_penalty` are not part of it.
- **The cards' values:** Qwen3.8-27B and Qwen3.8-Flash think at temperature 1.0, top_p 0.95, top_k 20, and answer
  without thinking at 0.7, 0.80, top_k 20 with presence 1.5; the Qwen3.8-35B-A3B distill thinks at 0.6 and uses the
  same instruct row; MiMo-V2.6-Flash, GLM-5.3-Flash and DeepSeek-V4.1-Flash use 1.0 and 0.95 (GLM has no official
  instruct mode). Fields a card leaves out are switched off; other models keep the library defaults.
- **`chat_template_kwargs`** takes `enable_thinking` and `reasoning_effort` in the form the Qwen cards use (the
  top-level field wins), and **`ie capabilities`** reports a `recommended` block (per mode the values, a when-to-use
  note and the maximum output; the effort levels, context and card link) and the real per-model `defaults`.
- DeepSeek-V4.1 and MiMo-V2.6 sample on the host without presence or frequency penalties: a nonzero value is served
  without them and logged as a WARNING. `scripts/hermes-engine` no longer wipes its own prompt-cache caps at
  `--parallel` > 1.

[Server controls](SERVER_CONTROLS.md).

### MiMo-V2.6-Flash sees, keeps conversations, and reports its vital signs

- **Native vision from the checkpoint's own tower**, on the GPU. The tower's block weights stream from pinned host
  memory, so an image encode borrows a 766 MiB block on the first card and returns it.
  Keeping that block free costs about 1.3 % of decode time in an A-B-A on the deterministic expert path.
  Its output sits within 5.4e-3 relative L2 of an fp32 reference of the checkpoint's own module, 11–20× closer than
  the module's own BF16 run.
  Images go in as OpenAI `image_url` parts, a picture the prompt cache has seen is never encoded again, and
  `GET /props` says whether this load takes images.

- **Other conversations are kept in host memory.** Switching back to a conversation restores its state instead of
  re-reading it: a 123.7K-token conversation resumed after ten side requests in 1.7 s instead of 415.3 s.
- **A discarded reply resumes in seconds.** The state at the end of the prompt is kept, so a request that repeats the
  prompt after a cut or discarded reply starts decoding at once: 3.9 s instead of 44.8 s to the first token at 16K
  tokens, bit-identical to a fresh read.
- **Tool calls repaired.** At temperature 1 the model sometimes leaves a parameter unterminated or quotes an enum value;
  each call is now parsed on its own, repaired where the intent is unambiguous, and logged. The host sampler sorts only
  as far as the top-p nucleus reaches: 12 ms to 1.8 ms per sampled row with the same picks.
- **Vital signs, opt-in.** A streamed request with `"ie_vitals": true` gets the sampling distribution's entropy and
  margin and the drafter's hit counts every 16 tokens, and a summary at the end. Without the field the stream is byte
  for byte what it was;
  with it, sampling costs about 0.1 ms more per row.
  MiMo-V2.6 only for now.

[Conversations in host memory and the prompt-end snapshot](docs/mimo26/P7_FIX64_FIX70.md), [vital signs](docs/mimo26/IE_VITALS.md).

### Serving that holds up

- **DeepSeek-V4.1's disk prompt cache survives rebuilds.** Entries are keyed by a manifest of the source files that
  decide its numerics, not by the executable, so an unrelated rebuild no longer empties the cache; a unit test checks
  that no numerics source is missing from the key.
  The first turn after a rebuild restored from disk in 1.4 s where the cold prefill had taken 79 s.
- **An idle-spin watchdog.** If the server burns CPU with nothing in flight, it names the busy threads in its log and in
  `/health`;
  20 of 20 client disconnects mid-generation left it idle.
- **A lost device ends the process** (exit code 75) so a supervisor can see it, instead of leaving an unhealthy server
  spinning in teardown; `IE_EXIT_ON_DEVICE_LOST=0` keeps the old behaviour.
- **Steadier long sessions:** `ie serve` grows less per request (glibc's mmap threshold is pinned) and no longer
  leaks an expert-cache event per reply, the CPU expert path also serves DeepSeek-V4.1 prompt chunks, and teardown
  drains every in-flight copy before it frees memory.

[The disk cache key](docs/deepseek41/103_PREFIX_CACHE_NUMERICS_KEY_2026-09-24.md), [the idle-spin watchdog](docs/server_idle_spin_watchdog_2026-09-24.md).

### A newer Intel driver stack

- **Released on Intel compute-runtime 26.35, IGC 2.41.5 and GuC firmware 70.65** (from compute-runtime 26.22, IGC 2.33
  and GuC 70.54). A SYCL context that holds both cards no longer costs host RAM for device memory: allocating 8 GiB on
  one card through such a context now leaves available host RAM unchanged, where it used to drop by 8.03 GiB. The engine already worked around that mirror with one
  context per card ([below](#earlier-updates)).
- **Output quality is unchanged.** DeepSeek-V4.1 perplexity 1.88767 against 1.88776 on the old stack over 16,383 tokens
  (0.04 standard errors apart), MiMo-V2.6 wikitext-2 3.4734 against 3.4749, and the Qwen3.8-27B gives byte-identical
  text on both stacks with the same binary. Greedy replies of the MoE models can differ from the old stack's at near-ties
  (new code generation). No GPU faults over the trial; the card firmware (IFWI) was checked, not changed.
- **Qwen3.8-27B `--spec`** passes its 5-prompt byte-identity certification (twice, deterministically) on the new stack,
  but it is not byte-identical to plain greedy decoding on every prompt: on one of two longer served prompts it picked a
  different near-tie token.

### Measured and closed

- **A restricted draft vocabulary for MiMo's drafter** (scoring only the most frequent token ids) is built, measured and
  left out of this release: the drafter's LM head is 0.74 % of decode time, and a 64K-token subset would make 2.6 % of generated tokens
  impossible to draft. [Measurement](docs/mimo26/P4_B5V_DRAFT_VOCAB.md).

---

## Highlights

- 🐋 **DeepSeek-V4.1-Flash on two B70s** — the 475 GB safetensors checkpoint with host-resident experts: 223K-token context verified, native tool calls and **image input** through `ie serve`, a prompt cache that answers follow-up turns in ~1–2 s, and prompt-lookup speculation that decodes agent tool loops **1.47× faster**, and several requests at once ([lanes](#several-requests-at-once-request-lanes)). See [the V4.1 numbers](#deepseek-v41-flash).
- 🆕 **MiMo-V2.6-Flash** — Xiaomi's 309B / 15B-active hybrid sliding-window model, running from its safetensors the day it was released: XMX attention kernels for its 192/128 head sizes, FP8 dense weights kept FP8 on the card, its bundled DFlash drafter decoding **1.46×** faster, native vision, conversations kept in host memory, and request lanes that decode several requests together ([lanes](#several-requests-at-once-request-lanes)). See [the MiMo numbers](#mimo-v26-flash).
- 🧩 **Several models, one endpoint** — `ie supervise` routes each request by model name to its own `ie serve` child on its own cards; `--cards`, layout files and `ie cards` make placement explicit. See [New in v0.2.0](#new-in-v020).
- 🏛 **Dense, MoE and hybrid models** — GLM-5.3-Flash, DeepSeek-V4.1-Flash, MiMo-V2.6-Flash, DeepSeek-V4-Flash, Qwen3.8-Flash-Next, Qwen3.6, Qwen3 / Coder / Tongyi, Qwen3-Next, Gemma-4, gpt-oss, and Llama-compatible dense models. GLM-5.2 and Tencent Hy4-preview have experimental standalone runners. See [architecture coverage](#supported-architectures) for entry points and status.
- 👁 **Native vision** — DeepSeek-V4.1-Flash and MiMo-V2.6-Flash (each checkpoint's own vision tower, no extra files), Qwen3.8-Flash-Next and experimental DeepSeek-V4-Flash-Vision-Exp, including image inputs through the OpenAI-compatible API. DeepSeek-V4 vision requires its native vision sidecar weights.
- 🥇 **Beats llama.cpp on Arc** — on prefill *and* decode across the models below.
- 🧠 **Runs the big ones** — gpt-oss-**120b** (117B) and Qwen3-Next-**80B** on 2× B70 via tensor-parallel; **~2.5× faster than LM Studio** on 120b.
- 🔀 **Multi-GPU built in** — `ie serve --gpus 2` (tensor-parallel + layer-split), no P2P required.
- 🔌 **OpenAI-compatible server** + tool-calling (Harmony + Qwen) + **image inputs** (`image_url` data URIs) — point any OpenAI client (or [Hermes](https://github.com/NousResearch)) at `:11435`.
- 📦 **One-command Docker** — `docker pull` (or build) → `ie-docker serve` → running on your Arc GPU in minutes. The prebuilt image dates from July 2026 and predates DeepSeek-V4.1, MiMo-V2.6 and `ie supervise`; build from source for those.
- ✅ **Correctness-first** — PPL-validated, per-layer cosine ≈ 1.0 vs a llama.cpp oracle, bit-exact where claimed.

---

## Supported architectures

Coverage below describes this source branch. A standalone runner does not imply
integration with `ie serve`; experimental ports have narrower qualification than
the benchmarked models. Prebuilt container images may lag these source updates.

| Family / GGUF architecture | Models | Entry point and coverage |
|---|---|---|
| **GLM-5.3-Flash** · `glm5next` | UD-Q4_K_XL GGUF | `ie serve` and `ie-glm5next-run`; sparse MLA + KDA, host-resident MoE, two-GPU pipelined prefill and MTP draft; kernel and full-model validation in [PERFORMANCE.md](PERFORMANCE.md); request lanes and a two-card lane pipe in a test tool (`ie-glm-lanes-test`, off by default; `ie serve` runs one request at a time) |
| **DeepSeek-V4.1** · safetensors directory (`deepseek_v41`) | Flash (FP8 dense, MXFP4 experts) | `ie serve <model dir>` and `ie-ds41-run`; two-card pipeline, CSA/engram/hyper-connections, three expert tiers (VRAM, pinned host RAM, NVMe), chunked long-context prefill, native DSML tool calls, native vision, prefix cache (memory + disk, keyed by its numerics), prompt-lookup speculation, request lanes and a two-card decode pipe served with `ie serve --parallel N` (both cards); needs ~256 GB of system RAM. |
| **MiMo-V2.6** · safetensors directory (`mimo_v2`) | Flash-RL (FP8 dense, MXFP4 experts) | `ie serve <model dir>` and `ie-mimo26-run`; two-card pipeline, 9 full-attention + 39 sliding-window layers with sinks (K 192 / V 128) on XMX prefill and split-K decode kernels, FP8-resident dense weights, three expert tiers (VRAM, pinned host RAM, NVMe) with a CPU expert path, the checkpoint's DFlash drafter for speculative decoding (on by default; prompt-lookup speculation when a checkpoint has none), XML tool calls, thinking on/off, native vision (the checkpoint's own tower), conversations kept in host memory and a prompt-end snapshot, request lanes (card-pipelined, row-batched) served with `ie serve --parallel N`, opt-in vital signs; measured with 256 GB of system RAM. Pro not yet qualified |
| **DeepSeek-V4** · `deepseek4` | Flash (ggml-org MXFP4, Q8_0 dense), Flash-Vision-Exp | `ie serve`; streaming expert caches, long-context sparse attention, prompt caching and structured tool calls; experimental native vision requires sidecar weights |
| **Qwen3.8-Flash-Next** · `qwen4exp` | Qwen4 preview | `ie serve`; DeltaNet + sparse QSA, hyper-connections, PLE embeddings, streamed MoE and native vision; request lanes on two cards served with `ie serve --parallel N` (up to 16, row-batched, shared-prefix cache) |
| **Qwen3.5 / Qwen3.6 / Qwen3.8 hybrid** · `qwen35`, `qwen35moe` | 27B dense (incl. Qwen3.8-27B), 35B-A3B MoE | `ie serve`; gated-DeltaNet + full attention, dense or MoE feed-forward paths; the 27B and 35B-A3B MoE splits serve request lanes with row batching and a shared-prefix cache (`--parallel N`, up to 16; the cache is opt-in on the 27B) |
| **Qwen3 MoE** · `qwen3moe` | Coder-30B-A3B, Tongyi-30B | `ie serve`; QK-normalized attention and routed MoE |
| **Qwen3-Next** · `qwen3next` | 80B-A3B | `ie serve`; DeltaNet + full attention and 512-expert MoE |
| **gpt-oss** · `gpt-oss` | 20b, 120b (MXFP4) | `ie serve`; attention sinks, sliding-window attention, Harmony chat and tool calls |
| **Gemma-4** · `gemma4` | 31B dense, 26B-A4B MoE | `ie serve`; per-layer head geometry, sandwich norms, softcap and sliding-window attention |
| **Qwen dense** · `qwen2`, `qwen3` | Qwen2/2.5/3, compatible Qwen distills | `ie serve`; shared dense transformer path with architecture-specific attention handling |
| **Llama-compatible dense** · `llama`, `phi3`, `granite` | Llama-3.x, compatible Mistral, Phi and Granite GGUFs | `ie serve`; shared dense path; compatibility depends on GGUF architecture and tensor layout |
| **GLM-5.2** · `glm-dsa` | GLM-5.2 | Experimental `ie-glm52-run`; standalone MLA + MoE forward path |
| **Tencent Hy4-preview** · `hyv4` | Hy4-preview | Experimental `ie-hyv4-run`; standalone generation/PPL, two-GPU stage split and STQ1/IQ1 quantization support; no general server qualification claimed |

Every model `ie serve` runs can also be one server of an `ie supervise` layout, on the cards the layout gives it
([layouts](docs/serve_config.md)).

**Recognized without an inference runtime:** Inkling-Small (`inkling`) and
Laguna S 2.1 (`laguna`). Loader recognition is not runnable model support.

**Weight import:** `ie import` converts supported AWQ, GPTQ and EXL3 safetensors
to native GGUF. Import format support does not add an unsupported architecture.

---

## Earlier updates


<details>
<summary><b>September 22 back to September 10, 2026</b> — MiMo-V2.6-Flash on release day, DeepSeek-V4.1-Flash vision and prompt-lookup speculation, chunked long prompts, the two-card memory root cause</summary>

**September 22, 2026 — MiMo-V2.6-Flash on release day.**

- **Attention kernels for MiMo's head sizes.** Its 9 full-attention layers use 192-wide keys and 128-wide values with
  per-head sinks, outside the engine's existing fast attention kernels. An XMX (joint_matrix) prefill kernel and a split-K decode kernel
  written for those sizes took a 32K prompt from 34.9 to 241 tok/s and 32K decode from 571–680 to 101–105 ms/token;
  with the router, sliding-window and chunking work after them, 32K prefill runs at 434–438 tok/s.
- **Serving defaults, each measured A-B-A.** A residency ranking from chat traffic (−16 % per decode token), a static
  expert tier sized from free VRAM (−11 %), FP8 dense weights kept FP8 on the card (−8 %, perplexity unchanged) and
  prompt-lookup speculation (−13 % on agent turns): 16.1 tok/s decode on held-out Dream prompts, 16.8 tok/s in a Dream
  agent loop. Wikitext-2 perplexity 3.4749 against llama.cpp's 3.4794 on the same checkpoint.
- **DFlash speculative decoding, from the checkpoint's own drafter.** MiMo ships a 5-layer block drafter that reads the
  model's hidden states and proposes 7 tokens at once; the model verifies them in one multi-row step. Drafts the
  drafter rates below 0.7 probability are not sent (an extra verify row costs ~27 ms). Held-out Dream prompts decode
  at **23.6 tok/s** (42.1–42.7 ms/token) against 16.2 plain and 18.1 with prompt lookup, the previous default; verbatim
  copies run as fast as lookup's. The drafter's drafts match a CPU fp32 reference on 1,736 of 1,736 tokens.
  [Measurements and limits](docs/mimo26/MIMO_V26_FLASH_2026-09-22.md).

**September 18, 2026 — DeepSeek-V4.1-Flash sees images, and agent loops decode 1.47× faster.**

- **Native vision from the checkpoint's own tower** (32-block ViT + aligner, 926 MB of weights already in the
  shards): images go in as OpenAI `image_url` parts, including a tool's screenshot. The tower's rows match DeepSeek's
  reference implementation at rel-L2 0.1–0.6 % (the model's own bf16 run is at 5 %), bit-identical run to run. The
  encoder borrows 1.5 GB of VRAM only while an image encodes (0.2 s for 640×480, 2.9 s at the 1,024-token budget),
  and a follow-up turn about the same image is served from the prompt cache without re-encoding it.
  [Tower](docs/deepseek41/94_PHASE55_VISION_TOWER_2026-09-18.md), [in the forward](docs/deepseek41/95_PHASE56_IMAGE_POSITIONS_IN_THE_FORWARD_2026-09-18.md),
  [serving and a 42-step agent soak](docs/deepseek41/96_PHASE57_IMAGES_IN_SERVE_AND_THE_AGENT_SOAK_2026-09-18.md).
- **Prompt-lookup speculation.** Agents spend much of their output copying text they have already seen (a todo list
  rewritten with one change, a file written back, a quoted string). When at least 12 context tokens match, the engine
  drafts the next 7 from that earlier occurrence and verifies all of them in one forward; prose is never drafted. The
  first A/B gave only 1.04×: an 8-row verify sent 17 GB of experts over PCIe, because the CPU expert path served
  one-row steps only. With that path extended to multi-row steps the verify drops 780 → 488 ms, and 42 recorded
  agent requests (18–34K context, replayed byte-identically, A-B-A) decode at **106 → 71.6 ms/token (9.4 → 14.0
  tok/s, 1.47×)**, outputs bit-identical to plain decoding with the CPU split off. On by default in `ie serve`
  (`IE_DS41_LOOKUP=0` turns it off). [Lookup speculation](docs/deepseek41/97_PHASE58_PROMPT_LOOKUP_SPECULATION_2026-09-18.md),
  [multi-row CPU expert path](docs/deepseek41/98_PHASE59_MULTI_ROW_CPU_EXPERT_LEG_2026-09-18.md).

**September 16–17, 2026 — DeepSeek-V4.1-Flash: faster long prompts, tool calls in the server, a prompt cache.**

- **Chunked prefill with the two cards as pipeline stages.** Card 1 works on chunk *k* while card 0 works on chunk
  *k+1*, so both PCIe links stream expert weights at once (a 2,048-token chunk moves ~180 GB of expert bytes).
  Bit-identical to the serial loop. With a gathered continuation attention (each row reads only its ~640 live keys)
  and candidate-block indexer scoring: **32K prompt 165 → 319 tok/s, 24K real text 183 → 344 tok/s, 223K real text
  104 → 319 tok/s** (36 → 12 minutes, needle at 50 % depth still retrieved). A split top-k defect that aborted every
  decode step past 131K keys was found and fixed on the way. [Pipeline](docs/deepseek41/83_PHASE43_PIPELINED_CHUNK_PREFILL_CRITERIA_2026-09-16.md),
  [gathered attention](docs/deepseek41/84_PHASE44_GATHERED_CONT_PREFILL_ATTENTION_CRITERIA_2026-09-16.md),
  [candidate skip + 223K](docs/deepseek41/85_PHASE45_INDEXER_SCORE_CANDIDATE_SKIP_CRITERIA_2026-09-16.md).
- **V4.1 behind the OpenAI server with native tool calls** (the DSML format, checked against the checkpoint's own
  encoder: 654 checks, shipped examples byte for byte), and prompts longer than one 2,048-token forward admitted.
- **Prefix cache.** Small ring checkpoints at every chunk boundary (the latent caches are append-only), other
  conversations kept in host memory, and a client's system prompt + tools written to disk once. A Dream-style chat
  with 90 tool schemas (17.7K tokens): time to first token **68.5 s → 1.1–2.3 s** on later turns, **2.4 s** on the
  first turn of a new server process. Every restore is bit-identical to recomputing (21/21 checks).
  [In memory](docs/deepseek41/86_PHASE46_PREFIX_CACHE_2026-09-16.md), [on disk](docs/deepseek41/87_PHASE47_DISK_PREFIX_CACHE_2026-09-16.md).
- **A served-process thread leak fixed.** Every NVMe expert fill spawned a thread whose OpenMP team was never
  reclaimed: 94K threads and ~0.5 GB per reply until decode starved. Persistent readers keep a server flat
  (~150 threads). [Details](docs/deepseek41/88_PHASE48_READER_THREAD_LEAK_AND_VRAM_2026-09-16.md).
- **Chat decode.** An expert-residency ranking profiled on chat decode (held-out prompts **7.26 → 10.81 tok/s**,
  A-B-A confirmed), engram rows faulted in parallel (host prep 9.4 → 2.7 ms/token), and an NVMe expert file rebuilt
  for that ranking (decode −6 % ms/token, prefill +6 %). A decode token is now expert bytes on PCIe, NVMe and
  E-core DRAM plus ~17 ms of attention. [Ranking](docs/deepseek41/89_PHASE49_CHAT_DECODE_RANKING_2026-09-16.md),
  [decode terms](docs/deepseek41/90_PHASE50_51_CHAT_DECODE_TERMS_2026-09-16.md), [expert file](docs/deepseek41/91_PHASE52_CHAT_EXPERT_FILE_2026-09-17.md).
- **New V4.1 defaults (September 17).** 10 GiB more pinned expert memory (still leaving at least 30 GiB free at load),
  the XMX W4A16 prefill expert route, and an FP8 LM head quantized at load into the checkpoint's own FP8 layout
  (perplexity unchanged: −0.0007 nats paired, 99.46 % top-1 agreement; +4 % decode, 0.66 GB of VRAM returned to
  the expert cache). 2K-token prompts **291 → 304 tok/s**. Each has an off switch (`IE_DS41_PIN_EXTRA_GIB=0`,
  `IE_DS41_PREFILL_XMX=0`, `IE_DS41_HEAD_FP8=0`). [Measurements and gates](docs/deepseek41/93_PHASE54_PIN_XMX_FP8_HEAD_DEFAULTS_2026-09-17.md).

**September 11, 2026 — two-card memory root cause, and the standard DeepSeek-V4-Flash GGUF.**

- **Every two-card run was mirroring its VRAM into host RAM.** On this stack a device allocation made in a
  SYCL context that contains both B70s (including the platform default context that `sycl::queue(device)`
  binds to whenever both cards are visible) is made resident on the peer card, and the kernel driver keeps a
  system-memory copy: 20 GiB of `malloc_device` cost 20 GiB of `MemAvailable`. The allocator, the DeepSeek
  runtime and the fleet's pipeline-split paths now use single-device contexts. GLM-5.3-Flash at 16K context
  pins every stage-1 expert layer for the first time: decode **9.9 → 14.5 tok/s**, prefill **66–106 → 156 tok/s**,
  identical text; the Qwen3.8-27B split takes **1 GiB** of host RAM at load instead of 29 at the same speed;
  DeepSeek two-card runs leave **0 GiB unattributed**. Tensor-parallel fleets keep their shared context on
  purpose (per-device contexts drop 27B tensor-parallel prefill 503 → 280 tok/s).
  [Root cause, mechanism and the runs](docs/glm53/DECODE_HOST_WAITS_2026-09-10.md).
- **ggml-org's DeepSeek-V4-Flash-0731 MXFP4 GGUF (the non-abliterated model) loads and runs at full speed.**
  Its 661 dense tensors are Q8_0; five F32-consumed roles now bind from Q8_0, and Q8_0 dense projections take
  the tuned Q8-SoA / oneDNN route instead of the packed per-element kernels: prefill **66 → 571 tok/s** at
  4,096 tokens, decode **18.9 → 26.1 tok/s** at 4K context, batch perplexity 3.948 on the first 512 wikitext-2
  test tokens. [Dtype map, change and measurements](docs/deepseek4/80_GGML_ORG_Q8_DENSE_BIND_2026-09-11.md).
- **Measured and closed:** GLM's expert CPU/GPU split ratio is flat within noise once every bank is pinned
  (pipelined MTP draft = 1.15× over serial); removing four DeepSeek per-layer host syncs was correct on every
  NLL dump but did not move decode, which is bound by expert-miss DMA at 4K (patch archived);
  DeepSeek expert prefetch is a dead end on the two-card runtime; Flash-Next's opt-in P2P context costs 60 GiB
  of host RAM for ≤ 2% decode. [Host-sync inventory and profile](docs/deepseek4/79_DECODE_HOST_SYNC_PLAN_2026-09-10.md),
  [prefetch verdict and the parked CPU/PCIe miss-split plan](docs/deepseek4/81_DECODE_PREFETCH_VERDICT_AND_CPU_SPLIT_PLAN_2026-09-11.md).

GLM-5.3-Flash now uses faster expert top-k selection, B70 KDA recurrence and
shape-specialized attention indexing. In-model kernel buckets improved by
**1.61×, 1.33× and 1.24×**, respectively, in the measured 24K profiling run.
Shared Q8 projections, FP16 XMX tiles, MLA tile reuse and convolution state
handling also received performance or correctness improvements.

These are kernel gains. Repeated full-model comparisons did **not** establish
a combined throughput improvement; slower runs also showed more expert staging
and host I/O. See [performance and validation](PERFORMANCE.md) for the measured
results, rejected experiments, test coverage and reproduction commands.

</details>

---

## Benchmarks

**Re-measured for v0.2.0 on the new driver stack** (compute-runtime 26.35), September 27, 2026, on a quiet machine
(no other heavy work, checked before each run), on the same commands and prompts as the dated rows they repeat:

| model | workload | v0.2.0, new driver stack | previous stack (date) |
|---|---|---:|---:|
| MiMo-V2.6-Flash | 10 held-out Dream prompts, 128 greedy tokens, DFlash drafter (the default) | decode **23.1–23.9** tok/s; prefill **438–483** at 4–6K | 23.6; 429–472 (September 22) |
| MiMo-V2.6-Flash | the same prompts, plain decoding | **16.7–16.9** tok/s | 16.2 (September 22) |
| DeepSeek-V4.1-Flash | `ie serve --ctx 32768 --parallel 2`, two tool-calling prompts (239–274 tokens each), A-B-A | **25.1–25.8** tok/s together, **×1.68–1.69** (one request 14.9–15.3) | 24.92, ×1.73 (September 26) |
| GLM-5.3-Flash | 16K-token prompt | prefill **156.4**, decode **14.0–14.4** tok/s | 156, 14.5 (September 10) |
| GLM-5.3-Flash | 2 lanes in `ie-glm-lanes-test`, served configuration, A-B-A | **21.35** tok/s together, **×1.68** (one request 12.64 / 12.76) | 22.26, ×1.676 (September 26) |
| Qwen3.8-Flash | pipelined prefill, 4×1024 tokens, once warm | **495–502** tok/s | 467–468 (August 28) |
| Qwen3.8-Flash | speculative decode, K=3, lossless: chat / code | **37.1–38.2** / **44.4–44.6** tok/s | 35.0 / 41.8 (August 28) |
| Qwen3.8-Flash | `ie serve --parallel 2`, A-B-A | **59.0–62.6** tok/s together, **×1.94–2.06** | — |

One to three runs per figure; each range spans the runs. The MiMo and V4.1 figures are on the v0.2.0 build, which stages
this driver's small device-to-host copies through pinned memory; without that change they measured 11–20 % lower on this
stack, with byte-identical output either way.
Qwen3.8-Flash's pipelined prefill warmed up over several runs after the model loaded: 206 tok/s on the first, 272–453 on
the next three, then 495–502.

**Request lanes, 1 to 16 requests at once** (release-candidate build, new driver stack), measured September 27–28, 2026
with the engine's lanes gate: `ie serve --ctx 32768 --parallel 16` (the 27B `--ctx 16384 --slot-ctx 4096`), served
defaults, greedy, short prompts, long replies, an A-B-A run per N with the last request arriving while the others
decode; tokens per second summed over the requests while all N decode. One run per figure. The 35B-A3B-class
row is on a community fine-tune of that model.

| model | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B class, row batching | 60.7–60.9 | — | **178.8** (×2.94) | **249.2** (×4.10) | **310.2** (×5.11) |
| Qwen3.8-Flash, row batching | 25.9–28.5 | — | **74.7** (×2.75) | **97.7** (×3.47) | **118.4** (×4.24) |
| Qwen3.8-Flash, `IE_Q4E_ROWS=0` | 26.2–28.5 | — | 60.9 | — | 67.1 |
| Qwen3.8-27B, lanes with row batching | 17.1 | 32.9 | **60.0** | **88.4** | **108.0** |
| Qwen3.8-27B, joint-step path (`IE_QWEN35_LANES=0`) | 17.1 | 31.9 | 49.3 | 60.0 | 63.4 |
| MiMo-V2.6-Flash, DFlash drafter on (the default) | 19.3–20.5 | — | 22.8 | — | 24.7 |
| DeepSeek-V4.1-Flash (long tutorial task) | 12.4–13.5 | — | 22.6 (×1.76) | — | 25.7 (×1.92) |

Each Qwen3.8-Flash run's first solo arm was the slowest (25.9 and 26.2 tok/s); the ×-figures are the gate's, against
the mean of the solo runs around each arm. The 27B lanes match or beat its joint-step path at every N.
Agents that share a system prompt with 16 tool schemas, four arriving at once, first token: 35B-A3B class (13.1K-token
prefix) 2.9–3.1 s with the shared-prefix cache against 60.5 s without; Qwen3.8-Flash (8.7K) 4.2–4.5 s against 65.6 s;
eight at once 5.6–6.2 s and 6.2–9.0 s.

Unless a block says otherwise, the rows below were measured on the previous Intel driver stack (compute-runtime 26.22);
the Qwen3.8-Flash, Qwen3.6-35B-A3B-class and Qwen3.8-27B blocks marked "new driver stack" and the request-lanes
block below on compute-runtime 26.35 ([New in v0.2.0](#new-in-v020)).

📊 **[Interactive charts →](https://red-weasel.github.io/machx-inference-engine/benchmarks.html)** · all measured on **Arc Pro B70** hardware; gpt-oss rows are clean-box head-to-head with identical GGUFs.


**gpt-oss-20b, head-to-head vs llama.cpp** — same GGUF, same GPU (1× Arc Pro B70), llama.cpp on its *fastest* config (FlashAttention on):

| context | **Mach X** prefill | llama.cpp | speedup | **Mach X** decode | llama.cpp | speedup |
|---|---|---|---|---|---|---|
| 512  | **1795** t/s | 927 | **1.94×** | **58.3** t/s | 50.3 | **1.16×** |
| 2K   | **4147** t/s | 927 | **4.47×** | **57.4** t/s | 49.9 | **1.15×** |
| 4K   | **3428** t/s | 896 | **3.83×** | **55.6** t/s | 49.4 | **1.13×** |

Wins both axes at every context length, and stays flat as context grows. Clean-box, reproducible (`ie-bench` vs `llama-bench`).

**gpt-oss-120b** (117B, MXFP4) — 2× B70, tensor-parallel:
| metric | Mach X | LM Studio (same 2 cards) |
|---|---|---|
| decode | **~31 tok/s** (peak 32) | ~12.4 tok/s |
| fit | full MXFP4, display-safe | — |

Coherent Harmony chat (math / poem / factual + multi-turn) and function-calling tool use. Batched-prefill PPL 15.20, bit-identical to T=1.

**Qwen3.6-35B-A3B "crown"** (all-Q8_0, ~36 GB) — 2× B70 vs llama.cpp SYCL layer-split:
| axis | Mach X | llama.cpp | speedup |
|---|---|---|---|
| prefill | **963** t/s | 763 | **1.26×** |
| decode | **63** t/s | 42 | **1.49×** |

PPL 6.36. Hybrid gated-DeltaNet + 128-expert MoE — one of the hardest architectures to run correctly, let alone fast.

**Tongyi-DeepResearch-30B** (qwen3moe) — 2× B70 tensor-parallel, long context (~17K):
| axis | layer-split | tensor-parallel | speedup |
|---|---|---|---|
| prefill | 124 t/s | **291** t/s | **2.35×** |
| decode | 21 t/s | **27.4** t/s | **1.30×** |

**Gemma-4** prefill (sliding-window attention) vs llama.cpp: **2.03× @4K**, **1.91× @8K**, **1.58× @16K**.

**Qwen3.6-27B** dense vs llama.cpp SYCL: prefill **1.21×** (349 vs 288 t/s).

**Speculative decode** (self-drafting MTP head, byte-identical to greedy on the certification prompts): Gemma-4 **1.46×**, Qwen3.6-27B **1.47×**.

**DeepSeek V4 Flash Vision-Exp Abliterated** (MXFP4 experts, F16/F32 remaining weights, 164.70 GB GGUF), two B70 cards with expert tensor parallelism. Measured September 10, 2026 after seven further kernel passes:

| prompt/context tokens | prefill tok/s | decode tok/s |
|---|---:|---:|
| 128 | **143.3** | **33.3** |
| 512 | **288.5** | — |
| 4,096 | **574.0** | **27.9** |
| 16,384 | **546.6** | — |

Three fresh unprofiled runs per variant in A/B/B/A/A/B order; single stream, synthetic prompts, 128 generated tokens, 2,048-token prefill chunks and eight decode warmup tokens. Loading is excluded. No speculative decoding or clean reboot. These are text-only measurements for the Vision-Exp derivative.

The 4K and 16K prefill medians changed by **+1.59%** and **+1.32%** against the previous seven-pass candidate. All **511 per-token likelihoods matched exactly**, with perplexity **5.4721**.

Decode medians changed by -0.41% at 128 context and -0.17% at 4K, with overlapping run ranges.

Five passes were retained: attention Q preparation fusion, MoE gather/scatter index reuse, RMS input reuse and in-place RoPE pair dispatch. RoPE also rejects dimensions that previously caused an out-of-bounds write. Q8 quantization grouping and Q8-to-half vector loads were slower and were rejected; both gained regression tests. All 20 affected CTest entries passed on each card. Coverage is still limited to two B70 cards and one compiler/driver combination.

[New run ranges, profiles, hashes and reproduction](bench/deepseek4/2026-09-09/SEVEN_MORE_PASSES.md). [Previous seven-pass results](bench/deepseek4/2026-09-09/SEVEN_PASSES.md).

<a id="deepseek-v41-flash"></a>**DeepSeek-V4.1-Flash** (safetensors: FP8 dense, MXFP4 experts, 475 GB), two B70 cards with 256 GB of
system RAM (≈197 GB pinned expert arena, at least 30 GiB left free), measured September 16–17, 2026, single runs:

| workload | prefill tok/s | decode tok/s |
|---|---:|---:|
| 2,048-token prompt, one forward | **304** | — |
| 32,768 tokens, synthetic repeated text, 2,048-token chunks | **319** | **14.9** (64 tokens) |
| 24,193 tokens, held-out real document | **344** | 10.4 (120 tokens) |
| 223,237 tokens, held-out real document, needle at 50 % retrieved | **319** (700 s) | 4.7 (300 tokens) |
| Dream-style chat: 17.7K-token system prompt + 90 tool schemas, `ie serve`, 6 held-out prompts | first turn **281**; follow-ups served from the prefix cache (1.1–2.3 s to first token) | **12.8** (range 10.5–15.5) |
| agent loop: 42 recorded tool-calling requests with screenshots, 18–34K context, `ie serve` (September 18) | follow-ups from the prefix cache | **14.0** with prompt-lookup speculation (9.4 without) |
| image input: 640×480 / 1920×1080 screenshot (206 / 968 image tokens) | vision encode **0.2 s / 2.9 s** | — |

Perplexity 1.887 on 16,384 wikitext-2 test tokens (exact path). Decode at long context is bound by expert
residency (VRAM and host RAM against the checkpoint), not kernels. Design and measurement notes:
[docs/deepseek41](docs/deepseek41).

**DeepSeek-V4.1-Flash served with `ie serve --parallel N`** (same machine and checkpoint), measured September 26, 2026 on
a server started with `--ctx 32768 --parallel 2` (lanes of 32,768 positions) in the served configuration, prompt lookup
on, greedy, thinking off; the server's own count of tokens while exactly one or two requests decode:

| prompts | one request alone, before / after | 2 together | ratio |
|---|---:|---:|---:|
| two prompts with nothing to copy, 320 tokens each | 14.81 / 13.99 tok/s | **24.92** tok/s, 12.46 each | **×1.73** |
| one of the two a copy task (prompt lookup drafts), 256 tokens | 14.04 / 14.23 tok/s | 24.27 tok/s (6.5 s window) | ×1.72 |

Three requests decode no faster than two: 118.7 ms per lane step with three decoding, about 25 tok/s (one run). A
one-row step takes a median 67.8 ms with one request decoding and 76.8 ms per request with two. A request with images
prefills in its own turn while the others wait (6.5 s in the measured case). [Method, checks and limits](docs/deepseek41/P4_B6B_SERVE.md).

**DeepSeek-V4.1-Flash lanes in the `ie-ds41-lanes-test` tool** (same machine and checkpoint; other prompts and settings
than the served run above), measured September 26, 2026 in the configuration `ie serve` uses (CPU expert path, FP8 LM
head, chat ranking, NVMe expert file), context 16,384, 4 prompts over 2 lanes, 192 tokens each, A-B-A:

| arm | 2 lanes together | one request alone, before / after (same prompts) | ratio | whole run incl. prefill |
|---|---:|---:|---:|---:|
| plain decoding | **16.32** tok/s | 11.17 / 10.63 tok/s | **×1.50** | **×1.58** (43.3 s vs 66.1 / 70.3 s) |
| prompt lookup on, copy-heavy prompts | **18.09** tok/s | 12.81 / 13.62 tok/s | **×1.37** | ×1.25 (53.0 s vs 69.3 / 62.7 s) |

Over three runs, plain decoding measured ×1.33–1.58, depending on how the window is counted; with lookup on, the arms
generate different text and so draft differently, which makes those ratios less exact.
With the CPU expert path off, every lane's logits and tokens match its solo run, lookup verify steps included.
During plain decoding each card is about 51–58 % busy with one request and 84–92 % with two lanes.
One extra lane at 16K context costs 125.0 MiB of VRAM on card 0 and 45.0 MiB on card 1, out of the static expert tier.

<a id="mimo-v26-flash"></a>**MiMo-V2.6-Flash** (safetensors: FP8 dense, MXFP4 experts, 178 GB), two B70 cards with 256 GB of system
RAM (115 GB pinned expert tier; 75 / 70 experts per layer in VRAM), measured September 22, 2026, single runs:

| workload | prefill tok/s | decode tok/s |
|---|---:|---:|
| 10 held-out Dream prompts (7 of 4.1–6.0K tokens), 128 greedy tokens each, with the bundled DFlash drafter (the default) | **429–472** | **23.6** (42.1–42.7 ms/token) |
| the same prompts: plain decoding / prompt lookup (the previous default) | — | 16.2 / 18.1 |
| 3 copy-heavy prompts (repeat a passage, re-emit a file or JSON with a rename), 512 tokens, DFlash | — | **24.3** |
| 32K-token haystack, needles at 10 / 50 / 90 % retrieved 3/3 | **434–438** | — |
| ~120K-token haystack, needle retrieved (ctx 131,072) | **322–324** | 12.3–13.6 |
| Dream agent loop through `ie serve` (write code and tests, run, fix; follow-ups from the live caches), before the drafter | — | **16.8** (14.7–18.9) |

Wikitext-2 perplexity **3.4749** (8 × 2,048 tokens) against 3.4794 ± 0.077 for upstream llama.cpp on a BF16 conversion
of the same checkpoint, and 3.7557 against 3.7526 at 8,192 tokens. These rows were measured text-only, one request at
a time; vision and request lanes came later. [Measurements, method and limits](docs/mimo26/MIMO_V26_FLASH_2026-09-22.md).

**MiMo-V2.6-Flash, several requests at once** (request lanes; same machine and checkpoint), measured September 26, 2026,
A-B-A with plain decoding and the CPU expert path on, 128 tokens per request on held-out Dream prompts:

| lanes | solo, before / after | lanes together | ratio |
|---:|---:|---:|---:|
| 2, one lane per card (card pipeline) | 16.79 / 16.49 tok/s | **28.17** tok/s | **×1.69** |
| 2, the same on the row-group build | 16.85 / 16.99 tok/s | **28.07** tok/s | **×1.66** |
| 4, two lanes per step (row groups), defaults | 16.10 / 15.08 tok/s | **28.85** tok/s | **×1.85** |
| 4, independent re-run on the same binary | — | **29.98** tok/s | **×1.90** |

With the CPU expert path off (a slower but run-to-run exact setting), 2 lanes reach ×1.68 and 4 lanes ×1.84, and every
lane's tokens match its solo run.
[Lanes](docs/mimo26/P4_B1_LANES.md), [card pipeline](docs/mimo26/P4_B2_PIPELINE.md), [row groups](docs/mimo26/P4_B3_ROWS.md).

**MiMo-V2.6-Flash served with `ie serve --parallel N`** (same machine and checkpoint), measured September 26, 2026 on a
server started with `--ctx 32768 --parallel 4 --slot-ctx 16384`, CPU expert path on, greedy, thinking off, 256 tokens per
request, on four held-out Dream transcripts (5.4–6.0K tokens) each first read into its own lane; the server's own count
of tokens while all N requests decode, against the same prompts' drafted rate alone:

| requests at once | one drafted request alone (same prompts) | together, lanes drafting (the default) | ratio |
|---|---:|---:|---:|
| 2 | 22.76 tok/s | **28.38** tok/s, 14.2 each | **×1.25** |
| 4 | 22.46 tok/s | **28.46** tok/s, 7.1 each | **×1.27** |

With drafting off while several requests decode (`IE_MIMO26_DRAFT_BUDGET=0`), a separate run on the same prompts gave
23.41 and 25.18 tok/s together.
Each configuration ran once on a busy machine, so compare within a run. The drafts are highly accepted on these
transcripts (92–99 % of the offered drafts kept); text that drafts less well was not measured with the budget.
A 1.6K-token prompt arriving while three requests decode got its first token in 4.4 s (measured before the draft
budget).
[Method, checks and limits](docs/mimo26/P4_B5_DRAFT_BUDGET.md); [serving before the draft budget](docs/mimo26/P4_B4_SERVE.md).

**GLM-5.3-Flash, two requests at once** (request lanes in the `ie-glm-lanes-test` tool, off by default; `ie serve` runs
GLM one request at a time), measured September 26, 2026 on held-out agent prompts re-rendered in GLM's chat template,
A-B-A in one process:

| configuration | one request | 2 lanes together | ratio |
|---|---:|---:|---:|
| CPU expert path on (as served), ~2K-token prompts, 128 tokens | 13.28 / 13.27 tok/s | **22.26** tok/s | **×1.676** |
| CPU expert path off, 4 prompts, 48 tokens | 9.54 tok/s | 18.16 tok/s | ×1.90 |

Each card goes from 58–60 % busy with one request to 99.5 % with two lanes (CPU expert path on).
With the CPU expert path off, each lane's logits and tokens are byte-identical to its solo run, over 2 mid-run
re-admissions; that run's steady window is short (4.7 s), so its ×1.90 is indicative.
One extra lane at 32K context takes 244.5 MiB on card 0 and 270.3 MiB on card 1 out of the expert cache.
[Measurements and limits](docs/glm53/P4_B7_LANES.md).

**Qwen3.8-Flash served with `ie serve --parallel N`** (UD-Q4_K_XL),
new driver stack, measured September 27, 2026 on `--ctx 32768 --parallel 2`, greedy,
512-token cap, 36–40-token prompts, A-B-A with the client's decode rate:

| run | one request alone, before / after | 2 together | ratio |
|---|---:|---:|---:|
| 1 | 29.00 / 31.89 tok/s | **59.00** tok/s | **×1.94** |
| 2 | 28.96 / 31.90 tok/s | **62.63** tok/s | **×2.06** |

Over 16 more requests in 8 waves of 2, host memory grew 1 MiB and VRAM and threads stayed flat. An extra lane at 32K
takes 0.554 + 0.553 GiB out of the expert cache.

**Qwen3.6-35B-A3B-class split served with `ie serve --parallel N`** (`qwen35moe`, Q8_0, measured on a fine-tune of
Qwen3.6-35B-A3B), new driver stack, measured September 26, 2026 on `--ctx 32768`, greedy, 512-token cap, A-B-A:

| lanes | one request alone, before / after | together | ratio |
|---:|---:|---:|---:|
| 2 | 61.25 / 61.10 tok/s | **117.09** tok/s | **×1.91** |
| 3 | 60.85 / 61.14 tok/s | **117.24** tok/s | ×1.92 |

One run each, over short windows (3.5–5.5 s): the two cards are saturated at two lanes. An 8,792-token prompt prefills in
about 12.3 s alone (~715 tok/s); two arriving together each got their first token in 23.6 s.


**DeepSeek-V4-Flash-0731** (ggml-org MXFP4 GGUF: MXFP4 experts, Q8_0 dense, 155 GB, the non-abliterated model), two B70 cards with expert tensor parallelism, measured September 11, 2026 on the same lines as above:

| prompt/context tokens | prefill tok/s | decode tok/s |
|---|---:|---:|
| 128 | **141.6** | **32.7** |
| 512 | **319.5** | — |
| 4,096 | **571.1** | **26.1** |
| 14,612 (serve, 64 greedy tokens) | **463** | **25.3** (30.5 warm cache) |

Before the September 11 loader and dense-route changes the same file measured 73.9 / 92.9 / 66.4 tok/s prefill and 21.8 / 18.9 tok/s decode. Batch perplexity **3.9170** on the packed route and **3.9476** on the shipped route over the first 512 wikitext-2 test tokens (stream-mode 3.9514), run-to-run byte-identical; the 64-token greedy continuation is identical between routes. No external reference for this exact file exists on the test box, so these are consistency figures, not an oracle comparison. [Details](docs/deepseek4/80_GGML_ORG_Q8_DENSE_BIND_2026-09-11.md).

Earlier measurements remain available: [September 3 binary benchmark](bench/deepseek4/2026-09-09/REPORT.md),
[eight-token expert optimization](bench/deepseek4/2026-09-09/OPTIMIZATION.md), and
[indexer optimization](bench/deepseek4/2026-09-09/INDEXER_OPTIMIZATION.md).

**Qwen/GLM kernel update, September 10:** in-place Qwen RoPE measures **2.25–3.04×** at the tested larger prefill shapes; GLM KDA gate **1.09–1.25×** at selected small chunks; half-rounded convolution-to-float fusion **1.04–1.38×** at T128–1024/C8192. Empty convolution calls now preserve dependencies. All eight affected regression tests pass on both B70s.

Paired Qwen 27B continuations, Flash-Next's 75MiB layer-output capture and all 490 GLM likelihood values match exactly. These are shape-specific kernel gains; whole-model timings remain mixed. [Changes, rejected RMS variants, model checks, existing GLM repeat-request variation and reproduction](bench/qwen-glm/2026-09-10/REPORT.md).

**Seven additional kernel passes, September 10:** retained bounded improvements to DeltaNet L2 normalization, fused QKV preparation, float/half gate preparation, QKV conversion, Q/gate splitting, head repetition and GLM KDA normalization. Final production-linked measurements range from **1.08× to 6.93×** across the retained shapes. All **11 affected regression tests pass on both B70s**, including exhaustive half-value checks on the vector conversion path. Paired Qwen/GLM outputs remain exact. Warm model gains are modest; cold GLM throughput declined, and a Qwen first-request delay did not recur on repeat. [Dispatch limits, timing distributions, rejected variants and model validation](bench/qwen-glm/2026-09-10/SEVEN_MORE_PASSES.md).

**Qwen3.8-27B** (Q8_0) — 2× B70, recorded August 15–26, 2026:
| workload | figure |
|---|---|
| short-context decode, layer-split + speculative | **22.0 tok/s** |
| short-context decode, tensor-parallel + speculative | **24.5 tok/s** |
| decode at 18.7K prompt depth, layer-split | **12.9–13.0 tok/s** (plain / speculative) |
| pipelined prefill, 2K / 4K / 9K prompts | **945 / 851 / 731 tok/s** |

The decode rows use the August 26 serving matrix; the prefill rows come from
the August 15 campaign with different prompts. Layer-split retains prompt caching.
See the [dated measurements and source excerpts](bench/deepseek4/2026-09-09/REPORT.md#archived-qwen-measurements).

**Qwen3.8-27B on the new driver stack** (Q8_0, 2× B70 layer split), measured September 26, 2026:

| workload | figure |
|---|---|
| `ie serve`, one request, plain greedy decode | **17.0–17.2 tok/s** (short prompts) |
| `ie serve`, prefill of a 3,382-token prompt | **820 tok/s** |
| `ie serve --spec --spec-k 3`, one request | **24.5 / 27.8 tok/s** decode (two prompts); prefill **924 tok/s** at 3.4K |
| `ie serve --parallel 2`, two requests at once | **32.28 tok/s** together against 17.32 alone (**×1.86**) |
| perplexity, built-in corpus, streaming | **5.3419** |

The September 10 binary gives byte-identical text on the old and new stacks. `--spec` passes its 5-prompt byte-identity
certification twice on the new stack, but on one of two longer served prompts it picked a different near-tie token
than plain greedy decoding, so it is not guaranteed byte-identical on every prompt. One run per arm, short prompts.

**Qwen3.8-Flash-Next** (UD-Q4_K_XL, 104 GB, host-resident experts) — 2× B70, recorded August 28, 2026:
| axis | figure |
|---|---|
| speculative decode, K=3 | **35.0 tok/s chat; 41.8 tok/s code** (lossless-greedy) |
| warm pipelined prefill, 4×1024 tokens | **467–468 tok/s** |
| warm serving prefill | **~290 tok/s** |
| decode during sampled agent runs | **~17–22 tok/s** |

These are archived campaign results, not a fresh run of the current build.
Standalone benchmarks and sampled agent workloads use different settings;
the agent runs did not complete their task. See the
[benchmark close-out and agent-run observations](bench/deepseek4/2026-09-09/REPORT.md#archived-qwen-measurements).

Vision demo: a 1236×1343 terminal screenshot is read at 990 vision tokens with OCR-level detail ("VS Code terminal… session capture… segmentation fault…"), ~15 tok/s decode with the image in context.

> Methodology varies by row: the dated DeepSeek, MiMo and Qwen campaigns describe their workloads and limits above. Historical `ie-bench --prefill P --decode N` rows mirror `llama-bench -pP -nN`. Some non-gpt-oss figures predate the latest clean-box sweep; the gpt-oss head-to-heads are ledger-verified.

---

## Built & tested on

**2× Intel Arc Pro B70** — Battlemage (BMG-G31), 32 GB GDDR6 each (**64 GB total**), **608 GB/s** bandwidth, ~183 FP16 TFLOPS via XMX. oneAPI 2026.x / SYCL. All single- and multi-GPU benchmarks above are on this hardware.
Released on Intel compute-runtime 26.35.39758.10, IGC 2.41.5 and GuC firmware 70.65.0 with the kernel's `xe` driver.

---

## Quick start

**Docker** — pull the prebuilt image (or build it yourself), then serve any GGUF on your Arc GPU:
```bash
docker pull ghcr.io/red-weasel/ie-engine:latest && docker tag ghcr.io/red-weasel/ie-engine:latest ie-engine
# ── or build from source (~15 min):   docker build -t ie-engine .
./scripts/ie-docker pull llama8b                     # or any Hugging Face GGUF
./scripts/ie-docker serve /models/…/model.gguf --gpus 1
# → OpenAI-compatible server on :11435 (point any OpenAI client at it)
```
Full 5-minute path in **[QUICKSTART.md](QUICKSTART.md)**.
The prebuilt image dates from July 2026: it does not contain DeepSeek-V4.1, MiMo-V2.6, `ie supervise` or the other
v0.2.0 features above. For those, build from source (below).

**From source** (needs oneAPI 2026.x + an Intel Arc GPU):
```bash
source scripts/env.sh
cmake -S . -B build -G Ninja && cmake --build build -j
./build/src/ie pull llama8b
./build/src/ie serve <model.gguf> --gpus 1
./build/src/ie serve --help          # every flag, its default, and the environment variables that matter
```

Multi-GPU: add `--gpus 2` (VRAM-aware; tensor-parallel + layer-split). Runs models bigger than one card — e.g. Qwen2.5-72B or gpt-oss-120b across 2× B70.

**Several models on one endpoint** (new in v0.2.0): list the cards, write a layout, start the supervisor.
```bash
./build/src/ie cards                                 # the GPUs --cards can choose, without loading anything
./build/src/ie supervise --config two-models.json    # one endpoint; each request goes to the server its "model" names
curl -s localhost:11470/v1/models                    # every server, loading or ready
curl -s -X POST localhost:11470/admin/shutdown       # asks every child to stop; nothing is killed
```
```json
{
  "version": 1,
  "default": "chat",
  "port": 11470,
  "servers": [
    {"name": "chat",  "model": "~/models/model-a.gguf", "cards": [0], "port": 11471, "ctx": 16384},
    {"name": "coder", "model": "~/models/model-b.gguf", "cards": [1], "port": 11472, "ctx": 16384,
     "restart": "on-failure"}
  ]
}
```
A layout with one server works with `ie serve --config <layout.json>` too, and `--cards 1` pins any single server to
card 1. [Layouts, routing, card locks and limits](docs/serve_config.md).

**MiMo-V2.6-Flash server** (2× B70, ~256 GB system RAM, the checkpoint directory):
```bash
./build/src/ie serve /path/to/MiMo-V2.6-Flash-RL --ctx 131072 --vram-reserve-gib 1.5
./build/src/ie serve /path/to/MiMo-V2.6-Flash-RL --ctx 32768 --parallel 4 --slot-ctx 16384   # four requests at once
```
The checkpoint's DFlash drafter is on by default (`IE_MIMO26_DFLASH=0` turns it off); images are accepted when `/props`
reports vision ready.
With `--parallel N` (up to 16), lane 0 keeps `--ctx` and every other lane gets `--slot-ctx` positions (default 32,768),
taken from the static expert tier; a load whose lanes do not fit is refused with the numbers. The second line is the
configuration the served measurements used. Images are served at `--parallel 1` only.


**DeepSeek-V4.1-Flash server** (2× B70, ~256 GB system RAM, the checkpoint directory):
```bash
./build/src/ie serve /path/to/DeepSeek-V4.1-Flash --ctx 75000
```
The directory needs `engram_tables.json` + `engram_token_map.i32` beside the weights (generate once with
`tools/ds41_reference/engram_tables.py`, which calls the checkpoint's own `inference/engram.py`). Optional beside
it: an expert ranking (`ie_ranking_decode_chat.txt` — the chat ranking is in
[results/ds41-chat-ranking-2026-09-16](results/ds41-chat-ranking-2026-09-16)) and an NVMe expert file written by
`ie-ds41-expert-file`. Prefix entries are cached under `~/.cache/machx-ie` (`IE_DS41_PROMPT_CACHE_DIR=0` turns
that off); `IE_DS41_PROFILE_OUT=<file>` profiles your own traffic into a ranking.
`--parallel N` serves several requests at once ([request lanes](#several-requests-at-once-request-lanes)); the lanes were
measured with `--ctx 32768`, and with more than one lane the load refuses `IE_DS41_PROFILE_OUT`.

**Use it from an agent desktop:** [Dream Agent Harness](https://github.com/Red-Weasel/Dream-Agent-Harness) launches
and manages `ie serve` (model picker, GPUs, context, tuning) and talks to it over the OpenAI API, including tool calls.
Dream finds a checkout at `~/machx-inference-engine` on its own; set `DREAM_MACHX_DIR=/path/to/machx-inference-engine` for
any other location. New in v0.2.0: `DREAM_MACHX_LAYOUT=<layout.json> dream local` starts `ie supervise` with a layout,
and Dream's settings can send sub-agents, the verifier or the filer to another server of that layout. With its
`engine.parallel` setting, Dream starts MiMo-V2.6-Flash or DeepSeek-V4.1-Flash with `--parallel N` and runs the
sub-agents of one reply side by side on the lanes.

**GLM-5.3-Flash server and Dream controls**:
```bash
./build/src/ie capabilities <model-00001-of-00006.gguf>
./scripts/ie-run-guarded --mem 220G ./build/src/ie serve \
  <model-00001-of-00006.gguf> --gpus 2 --ctx 200000 --threads 8
```
The server streams expert weights from host RAM and budgets resident weights,
context workspaces and expert caches per physical GPU. Dream now offers
**Model → GPUs → Context → Model tuning**, with validated sampling, stop,
threading and overflow controls. See [server controls](SERVER_CONTROLS.md) for
supported options, the observed GLM qualification and remaining limits.

**GLM-5.3-Flash standalone runner** (two B70 cards, host-resident expert banks):
```bash
./scripts/ie-run-guarded --mem 220G ./build/tools/ie-glm5next-run \
  <model-00001-of-00006.gguf> --gpus 2 --ctx 32768 \
  --prompt "Your prompt" --ngen 128
```
Prefill uses chunks of 1024 tokens and overlaps the two stages automatically,
including long `--prompt` input. Generation of 64 or more tokens automatically
uses MTP pipedraft when the model has an MTP head. Short generation, PPL scoring,
prefill benchmarks, plain-decode profiling/logit dumps and expert-parallel experiments do
not automatically load the draft head. Use `--no-pipeline --no-pipedraft` for
serial comparisons, or `--pipedraft` to request drafting explicitly. Drafting
uses extra model memory; its benefit depends on draft acceptance. These options
apply to the standalone runner, not the server.
Exact continuation comparisons use `IE_G5_CPU_MISS=0`; normal q* CPU/GPU expert
routing can produce different continuations between the two schedules.
Warm prefill batches that exceed the expert cache use waves of at most 16
experts, leaving room to retain weights needed later in the chunk. Cold
batches, batches that fit, and decode keep their existing wave width. This
adds no GPU workspace; `IE_G5_PP_WAVE=0` restores the previous half-cache
wave schedule for comparisons.
For prefill kernel timings, set `IE_QUEUE_PROFILING=1` and use
`--ppl <corpus> --ppbench <chunks>`; the runner reports instrumented kernel
time by bucket and device. Device windows cover the whole profile, including
gaps between that device's chunks.

Shared Q8 projections use shape-aware decode workgroups and combine aligned
prefill batches of at least 128 tokens into one grid. Small FP16 XMX GEMMs
use bounded 64-row tiles; larger or unsupported shapes retain the original
128-row path. These dispatches also apply to other models using the same
operators. GLM MLA projections load ahead without changing their accumulation
order, and sparse attention reuses latent tiles across heads. See
[performance and validation](PERFORMANCE.md) for the measured results and limits.

---

## Under the hood

- **Quantized GEMV** — W4A8/W6A8/W8A8 int-dot kernels (dp4a) over SoA-repacked weights: read each weight once, decode in-register. Q4_K, Q6_K, Q8_0, Q5_K, MXFP4.
- **FlashAttention** — register-tiled SIMD inner loop (no XMX for attention, following the fastest llama-SYCL path), plus split-K decode, sliding-window, and attention-sink variants.
- **MoE** — expert-batched weight-stationary prefill + fused gate/up/down; oneDNN XMX GEMM for the large-M regime.
- **Multi-GPU** — head-sharded attention + expert-sharded MoE (tensor-parallel) with host-bounced all-reduce; layer-split for pure capacity (bit-identical to single-GPU).
- **Speculative decode** — self-drafting NextN/MTP head with batched int-dot verify, byte-identical to greedy on each model's certification prompts (not guaranteed on every prompt: see the Qwen3.8-27B note in [Benchmarks](#benchmarks)); the checkpoint's own DFlash block drafter on MiMo-V2.6; prompt-lookup drafts for agent copies.
- **Request lanes** — per-request caches carved from the static expert tier (or free VRAM); the two cards run different lanes' steps at once, and on MiMo-V2.6 lanes' rows share one forward where every shared operation is bit-identical row by row. Qwen3.8-Flash, the 35B-A3B class and the Qwen3.8-27B share one lanes module for admission, lane choice, the card pipe, cancel and shutdown; on those three the decoding lanes' one-token steps run as one grouped card step that reads the weights once (row batching, byte-identical per lane), and a shared system prompt is snapshotted once and restored into any lane (on the 27B only with `IE_QWEN35_SHARED_PREFIX=1`).
- **Supervisor** — `ie supervise` routes by model name over per-card `ie serve` children, with per-card locks, byte-exact stream passthrough and an orderly shutdown that never kills a child.
- **Vision** — the model's own 449M SigLIP-style ViT ported natively: XMX GEMMs + custom LN / h-w rope / bidirectional packed-attention kernels, numpy-oracle-gated to 2e-6; embeddings splice into the LLM with true 3-stream interleaved M-RoPE (bit-identical to text rope when no image is present).
- **P2P pipeline** — 2-GPU layer-split with device-to-device wide-state push and double-banked chunk pipelining; every transport certified bit-identical.

Per-model design and measurement notes are in **[docs/](docs)**.

---

## License

**Apache License 2.0** — see [LICENSE](LICENSE). Copyright © 2026 Red-Weasel.

Free to use, modify, and ship (including commercially). Apache-2.0's patent grant + retaliation clause protects you and downstream users.

## Support

☕ **Buy me a coffee.** -- Unemployed and extremely grateful for any support -- If Mach X saves you time — or you want to see more fast local inference on Intel Arc —
donations are welcome, one-time or monthly. All donations support the project.

[![Buy me a coffee on Ko-fi](https://img.shields.io/badge/Buy%20me%20a%20coffee-Ko--fi-FF5E5B?logo=ko-fi&logoColor=white)](https://ko-fi.com/redweasel)

**[ko-fi.com/redweasel](https://ko-fi.com/redweasel)**

Requests and suggestions are welcome — [open an issue](https://github.com/Red-Weasel/machx-inference-engine/issues).
