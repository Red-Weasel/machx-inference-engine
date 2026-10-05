# MachX server controls

The current source build connects the existing GLM-5.3 `glm5next` runtime to
`ie serve`. Architecture dispatch applies to compatible models of that family;
it does not depend on the displayed model name. New architectures, unsupported
tensor layouts or quantization formats still require engine implementation.

## Loading and capability discovery

```bash
source scripts/env.sh
./build/src/ie capabilities /path/to/model-00001-of-00006.gguf
./scripts/ie-run-guarded --mem 220G ./build/src/ie serve \
  /path/to/model-00001-of-00006.gguf --gpus 2 --ctx 200000 \
  --threads 8 --temp 0.7 --top-k 40 --top-p 0.95 --min-p 0 \
  --repeat-penalty 1 --repeat-last-n 64 \
  --presence-penalty 0 --frequency-penalty 0 --max-tokens 4096 \
  --thinking on --reasoning-effort max
```

`capabilities` reads model metadata without loading GPU weights. Its versioned
JSON advertises architecture support, sampling/load controls and feature limits.
It does not guarantee that the requested allocation or every tensor type fits.
Unsupported architectures fail before generic model dispatch.

For GLM, one or two Level Zero GPU stages hold resident weights and expert
caches. The server now enables host-bank pinning by default and sizes each GPU
cache from live usable VRAM after reserving weights and context/workspaces.
The usable VRAM bound retains the existing headroom for other allocations.
A cache too small for one token's selected experts fails before weight uploads. The weight reserve
comes from the loader's metadata-only pass, using the exact tensor destination
precision and stage ownership. Source GGUF precision alone is not sufficient:
some stored F16 tensors are uploaded as F32. This shared calculation fixes the
September 7 refusal where a 5.42 GiB estimate was below the loader's 5.46 GiB.
The safety guard remains enabled.

Host pinning shares one live RAM budget across stages, reserving 40 GiB plus
load/staging scratch and applying the existing 1.37 host-USM overhead allowance.
Each layer also checks current RAM before pinning. A zero budget remains zero;
allocation failure or insufficient budget leaves banks on mmap and reports
partial residency. It does not promise that all weights fit pinned on every host.
The server logs per-layer loading progress and per-stage actual pinned/mmap bank
bytes and GPU cache capacity. `/props.memory_residency` exposes those allocation
counters; memory-mapped bytes are reclaimable and may be read from disk.
GPU expert caches duplicate selected host weights and may include empty or
quarantined slots. Do not sum host and GPU cache counters as unique model bytes.

`IE_G5_PIN_BANKS=0` explicitly disables host pinning. `IE_G5_ECACHE_MB` requests
an explicit per-stage cache budget; oversized or too-small requests fail rather
than silently shrink. `IE_G5_PIN_MAX_GIB` caps each stage's pinning; zero disables
it. `IE_G5_PIN_FLOOR_GIB` and `IE_G5_PIN_OVERHEAD` retain their advanced meanings
and now require finite, bounded values. `capabilities.memory_policy` reports
these server defaults/overrides without allocating weights. Standalone GLM
research runners retain their older defaults.

The original mmap default came from commit `40c49b92` on August 29, 2026, which
recorded a request to avoid 200+ GB used RAM. Dream's server integration inherited
it. This September 6 server correction responds to the owner's current loading
request. Physical GPU counting still avoids counting OpenCL aliases twice.
Allocation checks can still fail if another process consumes resources later.

## Generation options

Server defaults apply when an HTTP request omits an option. Explicit request
values, including zero, override those defaults. Invalid values return an error.

Recommended sampling (P4 B20). For Qwen3.8-27B, Qwen3.8-Flash-Next,
Qwen3.8-35B-A3B-Distill, MiMo-V2.6-Flash-RL, GLM-5.3-Flash and
DeepSeek-V4.1-Flash the server fills `temperature`, `top_p`, `top_k`, `min_p`,
`presence_penalty` and `repeat_penalty` from the model card's values for the
request's actual mode (thinking or instruct). The order is: request value, then
CLI flag or `IE_SERVE_*` environment, then the recommendation, then the library
default. A temperature of 0 set by the request or the server (greedy) takes
nothing from the recommendation. `max_tokens`, `frequency_penalty` and the
performance settings (prefill chunk and the rest) are not part of it.
`ie capabilities` reports the resolved `defaults` and a `recommended` block
(modes, values, max output, context, when-to-use notes, card URL); the server
logs the filled values per request. `chat_template_kwargs.enable_thinking` and
`chat_template_kwargs.reasoning_effort` are accepted as aliases of the
top-level fields (the top-level field wins). DeepSeek-V4.1 and MiMo-V2.6 sample
on the host without presence or frequency penalties: a nonzero value is served
without them and logged as a WARNING.

| CLI option | Accepted behavior |
|---|---|
| `--temp` | 0–2; zero is greedy |
| `--top-k` | 0–1024; zero uses the 1024-candidate ceiling |
| `--top-p`, `--min-p` | Greater than 0 through 1; 0 through 1, respectively |
| `--repeat-penalty` | Greater than 0 through 10; 1 disables repetition penalty |
| `--repeat-last-n` | 0–512 recent prompt/output tokens; zero disables history penalties |
| `--presence-penalty`, `--frequency-penalty` | -2 through 2; additive penalties over that history |
| `--seed` | Unsigned integer; zero chooses a random seed |
| `--max-tokens` | Output limit; zero uses the remaining context budget |
| `--stop` | Repeat up to four times for literal stop strings |
| `--threads` | 1–1024; CPU expert work on advertised backends, not HTTP workers |
| `--prefill-chunk` | Positive prompt batch size; affects workspace allocation |
| `--parallel`, `--slot-ctx` | Request lanes and the context of each lane after the first; `--parallel auto` (the default since v0.2.6) lets the load pick the count; architecture restrictions apply |
| `--thinking` | `on` or `off`; selects the supported thinking template |
| `--reasoning-effort` | Exact model-specific level advertised by `capabilities` |
| `--no-prompt-cache` | Disables reuse on backends that support it |
| `--int8-kv`, `--spec`, `--spec-k` | Architecture-specific; check capabilities before selecting |

The sampler considers at most 1024 candidates, even with `top-k=0`. History
penalties use a bounded window rather than the complete conversation. The
server rejects context overflow. Dream can instead compact its visible history
before admission; this does not implement GPU KV shifting.

GLM chat supports reasoning and native function calls mapped to OpenAI replies.
Each request resets and prefills model state. GLM server prefix reuse, MTP,
INT8 KV, vision and deferred tool loading remain unavailable. Standalone GLM
pipeline/draft flags do not configure the server. CUDA projector, Jinja override,
tensor-split and reasoning-budget flags from llama-server are not accepted as
MachX options. Runtime generation errors produce HTTP/SSE errors.

Reasoning effort now reaches the actual prompt rather than being ignored or
hardcoded. GLM and DeepSeek-V4 support `low`, `high`, `max`; GPT-OSS supports
`low`, `medium`, `high`; recognized Qwen effort templates support `low`,
`medium`, `high`, `xhigh`, with `high` mapped to `xhigh` by that template.
Non-reasoning templates do not expose effort or an ineffective thinking toggle.
Unsupported model/effort combinations fail before GPU initialization at startup
or return HTTP 400 before streaming starts. `IE_SERVE_REASONING_EFFORT` sets a
server default; explicit CLI and request settings take precedence. Capability
reporting also reflects the reasoning/thinking environment defaults, including
the older Qwen `IE_REASONING_EFFORT` override.

On the Qwen thinking templates (Qwen3.8-Flash, Qwen3.8-27B, the Qwen3.6-35B-A3B class), a reply with thinking on is
split since v0.2.6: the text before the model's `</think>` is returned as `reasoning_content` and the text after it as
`content`, streamed (`delta.reasoning_content`, then `delta.content`) and non-streamed alike. Before, the reasoning
arrived in `content` and `reasoning_content` was empty. A reply cut off before `</think>` is all `reasoning_content`
with an empty `content`, the rule DeepSeek-V4 / V4.1, MiMo and GLM already follow. Stop sequences count from the start
of `content`. With thinking off nothing changes.

Models whose GGUF chat template teaches the `<function=` XML tool form (Qwen3.8-27B, the 35B-A3B class; read from
the template, not the architecture) are prompted with that template's own tools block, and their XML calls come back
as structured `tool_calls` with parameters typed by the request's schema. Calls are taken only from the text after
`</think>`; a reply that ends with a complete call and never closed its reasoning is accepted when its finish reason
is `stop`; a function name the request does not list is rejected. A complete call block in the answer is taken as a
call even when prose follows it. `IE_QWEN_TOOLS_JSON=1` restores the Qwen3 JSON tool preamble. Each examined reply
logs one `[chat] xml tools:` line with where the answer starts and how many calls were parsed.

GLM's vendor template always opens a thinking block. MachX's `--thinking off`
uses a local empty closed thinking block to prompt a direct answer; it is not
a native vendor on/off flag. Its effort choices are independently documented
in the [official GLM-5.3-Flash model card](https://huggingface.co/zai-org/GLM-5.3-Flash).

## Prompt cache: reply snapshots

On the trie-backed prompt caches (the multi-card fleet caches — Qwen3-Next,
crown split, gpt-oss TP, qwen3moe split/TP, Qwen3.6/3.8-27B split — and the
single-GPU crown cache) the engine can also snapshot the state at the end of a
completed reply, keyed by `prompt ++ reply` tokens, so the next turn of the same
conversation restores through the reply and prefills only the template tail
plus the new user message. It does so **only for plain (non-thinking) ChatML
models** such as Qwen3-Coder / Tongyi, where the history render of an assistant
turn (`<|im_start|>assistant\n` + content) equals the generation prompt plus the
reply. Thinking-capable Qwen templates (Qwen3.5/3.6/3.8) put a `<think>` block
in the generation prompt — empty when thinking is off — that the official
template never renders into history, so the next turn's tokens diverge right
after the assistant header and such an endpoint could never hit; the engine
skips the snapshot there (measured 2026-09-10 on the 27B split: turn-2/3 hit
depth 27/52 tokens with and without it), and the prompt-boundary snapshot keeps
serving those turns as before. Each reply snapshot is one more LRU endpoint
sized to its depth; `IE_PROMPT_CACHE_MAX_ENTRIES` caps the count and
`IE_NO_GEN_CACHE=1` disables reply snapshots. The single-endpoint caches
(Gemma-4, Qwen3.8-Flash-Next, DeepSeek-V4 host slots) and GLM (no prompt cache)
are unchanged.

With request lanes on the Qwen3.6-35B-A3B class (v0.2.6), the prompt cache is sized for the lanes: up to
max(12, 2 × lanes + 8) entries inside a per-card VRAM budget (each card's free VRAM after the lanes, less a 1.5 GiB
reserve), and an insert evicts least-recently-used entries before it allocates. A conversation's new snapshot
replaces that conversation's older ones (`IE_Q35MOE_CACHE_SUPERSEDE=0` keeps them), except the one at its last user
query's end, which is kept as an anchor so a new query after a tool loop restores up to there (`IE_Q35MOE_ANCHOR=0`
turns that off). Each lane also keeps a checkpoint of its own conversation and restores the next turn in place
(`IE_Q35MOE_STICKY_LANES=0` turns that off). `IE_Q35MOE_LANES_CACHE=0` returns to 12 entries with no budget;
`IE_PROMPT_CACHE_MAX_ENTRIES` and `IE_PROMPT_CACHE_VRAM_MIB` override the two limits. The load log prints the entry
count, the budget and each rule's state.

## Admission, health and shutdown

`ie serve` runs up to `--parallel N` generations at once. On
MiMo-V2.6-Flash, DeepSeek-V4.1-Flash, Qwen3.8-Flash, the Qwen3.6-35B-A3B class and
Qwen3.8-27B, N up to 16 request lanes decode concurrently; each model checks at load
whether N lanes fit and refuses a count that does not (README, "Several requests at
once: request lanes"). Every other architecture takes whole-generation FIFO turns.

Since v0.2.6 the default is `--parallel auto` (the same as leaving the flag out): the
load picks the count and logs one `[lanes] auto ...` line with the reason. The 35B-A3B
class and the Qwen3.8-27B on their two-card splits take the most lanes, up to 16, that
their free VRAM holds at the default `--slot-ctx` (the 27B also within its load-time
budget); MiMo-V2.6-Flash, DeepSeek-V4.1-Flash and Qwen3.8-Flash take 4 lanes at
`--slot-ctx 16384` (an explicit `--slot-ctx` wins); one card, `--int8-kv`, `ie run`, a
switch that turns the lanes off or refuses them, and every other architecture take 1.
An explicit `--parallel N` loads exactly as before. A layout file's `parallel` key
takes a number only: leave it out for the pick. `ie capabilities` still reports
`defaults.parallel` as 1.

At most `--max-queue M` further requests wait for
a slot (default 8, or the pick + 8 when the load picked the count); beyond that the server answers **HTTP 429** immediately
with `Retry-After: 1` and `{"error":{"code":"queue_full"}}`, so a burst can
never exhaust the HTTP worker pool. The pool is sized `parallel + max_queue + 4`
so `/health`, `/props` and the 429s themselves are always answered.

`GET /props` returns `default_generation_settings.n_ctx` (the server's context), `total_slots`
(the request lanes this load runs, the pick included) and, since v0.2.6, `slot_ctx`: the
positions each lane after the first holds (lane 0 holds `n_ctx`), 0 with one lane. A client
that runs sub-agents on the lanes can size each one's context from it.

`GET /health` returns `{"status":"ok","inflight":..,"queued":..,"parallel":..,"max_queue":..}`
with 200 while the device is healthy. With request lanes on the 35B-A3B class, Qwen3.8-27B
and Qwen3.8-Flash it also carries the lanes' counters; v0.2.6 adds `snapshots`, `drains` / `drain_ms` / `drain_max_ms`, `lookaheads`,
`short_bypass`, `short_guard`, `short_guard_wait`, `recuts`, `quota_holds` and `quota_guard`. v0.2.8 adds `turns_nodrain`
(serial turns that ran beside the lane pipe instead of pausing it), `turns_boundary_first` (mark or snapshot turns taken
ahead of waiting prepare turns) and, on the 35B-A3B class only, `pipe`: `{"groups": [...], "regroup_waits", "regroup_merges",
"regroup_timeouts", "regroup_wait_ms"}` — the decode groups formed so far by size (1 to 16 rows; `[]` until the first group
forms) and the decode regroup's counters. Since v0.2.8 `drains`, `drain_ms`, `drain_max_ms` and `paused_ms` count only the
turns that paused the pipe. If a forward ever reports a lost or reset
device (`DEVICE_LOST`, "device lost", `DEVICE_RESET` in the error text), the
server latches the fault: `/health` turns **503** `{"status":"unhealthy","reason":..}`
and every generation is refused with 503 `code:"device_lost"` until the process
is restarted. Nothing in the engine re-creates a lost context, so a supervisor
should restart on a 503 health check rather than keep sending requests.

`/health` also carries `"idle_spin":{"seconds":N,"cores":X}` from the idle-spin
watchdog. The watchdog samples the process's CPU every 5 s. When nothing has been
in flight, queued or admitted for a whole 60 s window and the process still burned
at least `IE_IDLE_SPIN_CORES` cores (default 2.0) over that window, it logs one
`[ie] idle spin:` line and repeats it every 10 minutes while the spin lasts. The
line names the busiest threads (usr/sys split, kernel wait channel, whether each
is an HTTP request thread) and the `eu-stack` command for their stacks.
`seconds` is how long the spin has lasted (0 when none); `cores` is the rate over
the latest idle window. `IE_IDLE_SPIN_WATCHDOG=0` turns the watchdog off (the
field is then absent). It never stops or restarts anything
(docs/server_idle_spin_watchdog_2026-09-24.md).

Every error body is JSON-escaped (`error_json`), including exception text.

A client that disconnects is detected during prefill (the engine probes the
callback between prefill chunks) and during decode (every fragment), for both
streaming and non-streaming requests, and also for gpt-oss (Harmony) streams,
which buffer their output and previously could not observe a closed socket.

`SIGTERM` and `SIGINT` stop the server in order: in-flight generations abort at
their next token, queued requests get 503 `code:"shutting_down"`, the listener
closes, `run_openai_server` returns and every destructor runs (pinned host
arenas, device memory). A second signal while stopping exits immediately.
`POST /admin/shutdown` (loopback only) does the same without a signal.

Since v0.2.6 a stop on the request lanes of the 35B-A3B class, Qwen3.8-27B and
Qwen3.8-Flash no longer waits for queued work: every request ends at once with
finish reason `abort`, the steps waiting for a card are dropped, and the process
waits only for the steps already on a card. A running 35B-A3B prefill piece stops
at its next layer; on the 27B and Qwen3.8-Flash a running piece finishes its card's
part. The log ends with `[ie] stopped in X.X s (waited for N in-flight GPU steps)`.
Measured on October 1, 2026, stop request to process exit: 0.43 s where the previous
build took 14.58 s (4 lanes mid-prefill), 1.08 s where it took 49.70 s (16 lanes, an
80K-token prompt mid-prefill), 0.45–0.53 s on the release's defaults, 0.82 s on the
27B and 3.08 s on Qwen3.8-Flash, with no GPU fault and a clean restart after the
35B-A3B stops. On v0.2.8 the 16-lane stop with an 80K-token lead and six workers
measured 0.45 s (October 2, 2026; a 512-row piece at 42K positions in flight, no GPU
fault, a clean restart). The bound is one layer of the running prefill piece and was measured
down to 43K tokens of depth. Give a server that time: killing a process that still
has work on a card can leave the GPU faulting. DeepSeek-V4.1-Flash and
MiMo-V2.6-Flash stop as before, at the next token.

Image inputs are refused with an error when the server runs `--parallel > 1`:
per-request image staging is engine-global and not slot-safe. (DeepSeek-V4.1-Flash
takes images on its lanes.) Since v0.2.6 a load of MiMo-V2.6-Flash or Qwen3.8-Flash
without `--parallel` has 4 lanes, so pass `--parallel 1` to serve images on them.

The Qwen3.8-27B and the 35B-A3B two-card splits (`--gpus 2`) take images at `--parallel 1`
with their vision projector: `--mmproj <mmproj-...-F16.gguf>` (the base model's F16 or the
abliterated 27B's BF16 projector; the tower loads onto the second card at load, before the
lanes are sized, and the 27B's `[budget]` line counts it), else `IE_MMPROJ` or an
`mmproj*.gguf` beside the model (an F16 first), loaded at the first image. `/props`
`vision.ready` says whether a load takes them. An image request runs without the prompt
cache and without `--spec`; its text turns are unchanged. The lanes (`--parallel` > 1, the
default load of both splits) refuse images with that reason.

## Request lanes on the 35B-A3B class: scheduling switches (v0.2.6, v0.2.8)

These are read once per process and apply to the `qwen35moe` two-card split with more
than one lane. The load log prints each rule's state. Measurements are in the README,
"New in v0.2.6" and "New in v0.2.8".

Since v0.2.8 the serial turns this class takes for a lane (a cached prefix's restore, the
shared-prefix mark, the conversation snapshot, the prompt's end, the snapshot of a reply that
ended with a stop) run beside the lane pipe: the other lanes keep stepping, and the turn waits
only for its own lane's steps. The prompt's last rows (the generation prompt) go through the
pipe as the prompt's last piece, ahead of the other prompts' queued pieces. A lost lane's
reset, and the snapshot of a reply that ended at its length limit, still pause the pipe. The load log says
`[qwen35moe] serial turns run BESIDE the lane pipe ...`, or `DRAIN the lane pipe
(IE_Q35MOE_TURN_DRAIN=1)` with the switch. Qwen3.8-27B and Qwen3.8-Flash keep the paused turns.

| Switch | Default | Effect of the other value |
|---|---|---|
| `IE_LANES_PREFILL_FIFO` | on for this class (off on Qwen3.8-Flash and Qwen3.8-27B) | `=0`: every waiting prompt's pieces at once, so agents that share a prompt each read it; `=1` turns it on for the other two |
| `IE_Q35MOE_FA2_TILE_MINCTX` | 512 since v0.2.8 (6144 in v0.2.6): a prefill piece whose last position reaches this takes the tiled attention kernel, a piece that ends below it the previous kernel; `--parallel 1` included. The load log prints `[qwen35moe_split] prefill attention: tiled (FA2 wide tile) from 512 positions, naive below ...` | `=6144`: v0.2.6's kernel choice (the same bytes for the same prefill pieces). The 27B's `IE_QWEN35_FA2_TILE_MINCTX` is a separate knob and unchanged |
| `IE_Q35MOE_NO_FA2_TILE` | unset | set to any value, `0` included: the previous kernel at every depth and its exact output |
| `IE_Q35MOE_TURN_DRAIN` | unset (v0.2.8): the serial turns run beside the lane pipe (above) | `=1`: every turn pauses the pipe and the prompt's last rows run inside the prompt-end turn, as in v0.2.6 |
| `IE_Q35MOE_REGROUP` | on (v0.2.8): before forming a decode group that would leave three or more groups rotating over the two cards, card 0 waits up to `IE_Q35MOE_GROUP_WAIT_US` (20000; 1–1,000,000) microseconds for the decode group landing on card 1, then forms from every lane queued. One or two decoding lanes never wait, and a prefill piece at the front of the queue never waits. The load log prints `[qwen35moe] decode regroup ON: stage 0 waits up to 20000 us ...`; the shutdown line's `groups by size` and `/health` `pipe` carry the counts | `=0`: no wait; the groups stay as the lanes' arrival aligned them. Same bytes either way |
| `IE_Q35MOE_LANE_PIPELINE` | on: a lane's next prefill piece enters card 0 while card 1 runs the piece before | `=0`: one piece of a lane at a time |
| `IE_Q35MOE_SHORT_FIRST` | on: a prompt with at most `IE_Q35MOE_SHORT_ROWS` (16384) rows left goes ahead of a long prompt's next piece, which waits for at most `IE_Q35MOE_SHORT_SLOTS` (2) short pieces or `IE_Q35MOE_SHORT_WAIT_MS` (10000) ms | `=0`: the FIFO window and arrival order alone |
| `IE_Q35MOE_MIX_CHUNK` | 512: the rows of a prefill piece while another lane is busy | `=<rows>`; `=0`: the plan's 8,192-row pieces (re-cut is then off) |
| `IE_Q35MOE_RECUT` | on: a prompt that started alone cuts its remaining pieces to the cap once another lane decodes or reads a short prompt | `=0`: it keeps its pieces |
| `IE_Q35MOE_DECODE_QUOTA` | 32: a long prompt's next piece waits until every decoding lane made this many steps, for at most `IE_Q35MOE_QUOTA_MAX_MS` (2000) ms | `=0`: no wait; a deep prefill is fastest and the other lanes decode slowly beside it |
| `IE_Q35MOE_STICKY_LANES`, `IE_Q35MOE_LANES_CACHE`, `IE_Q35MOE_CACHE_SUPERSEDE`, `IE_Q35MOE_ANCHOR` | on | `=0` each: see "Prompt cache" above |
| `IE_LANES_TRACE` | off | `=1`: one log line per card step (a prefill piece's or a decode group's rows, its time, the idle gap before) |

A prompt longer than one piece that arrives beside running lanes, or that is re-cut, is
read in other pieces than the same prompt alone and can give other bits. Prompts that
start and run alone, prompts of up to one piece, and `--parallel 1` are unchanged.

## Prefill, speculation and allocation switches (v0.2.13)

Each is on by default and read once per process; `=0` restores the previous path (the two `NO_` switches are set to
`1` to skip a test). Figures and conditions: the README's "New in v0.2.13".

| switch | what it controls |
|---|---|
| `IE_QWEN35_ATTN_GEMM`, `IE_Q35MOE_ATTN_GEMM`, `IE_Q4E_ATTN_GEMM` | a prefill chunk's attention as two matmuls and a softmax (27B split, 35B-A3B split, Flash-Next; Flash-Next up to `IE_Q4E_ATTN_GEMM_MAXCTX` = 65536 keys) |
| `IE_QWEN35_S8_PREFILL`, `IE_Q35MOE_S8_PREFILL`, `IE_Q4E_S8_PREFILL` | Q8_0 projections read in place by the prefill matmul |
| `IE_QWEN35_DN_SCAN`, `IE_Q35MOE_DN_SCAN`, `IE_Q4E_DN_SCAN` | the register-resident DeltaNet scan for prefill chunks |
| `IE_QWEN35_SPEC_STEP` | with `--spec` on the 27B split: speculation as a step of the normal decode loop (any sampler, the prompt cache, any prompt length); `=0` = the greedy-only path with its 4,096-token limit |
| `IE_QWEN35_SPEC_ROWS` | the speculative verify rows' attention in one pass |
| `IE_Q4E_PLE_PREFETCH` | Flash-Next reads its per-layer token table into RAM in the background at load; `=wait` makes the load wait for it |
| `IE_NO_ALLOC_ALIAS_TEST`, `IE_ALLOC_ALIAS_MIN_MIB` (64), `IE_ALLOC_ALIAS_STATS` | the page-alias self-test of every large device allocation; its size threshold; print its totals at exit |
| `IE_Q4E_NO_ALIAS_TEST` | Flash-Next's own test of its expert caches |

Diagnostics: `IE_QWEN35_PFPROF=1` / `IE_Q4E_PFPROF=1` print where a prefill's time goes; `IE_DEBUG_PICKS=1` prints
every sampled token.

## Lookup drafts and restart points (v0.2.18)

`--spec` turns the lookup drafts on (on the 27B together with its MTP draft head); without it decoding is unchanged.
The restart points are on by default with the prompt cache. Figures and conditions: the README's "New in v0.2.18".

| switch | what it controls |
|---|---|
| `IE_QWEN35_LOOKUP` | 27B split, one request lane: drafts taken from the conversation's own text; `=0` = the MTP head only |
| `IE_QWEN35_LOOKUP_FAST` (128) | the longest round checked through the prefill kernels; `=0` = exact rows only (text equal to plain decoding) |
| `IE_QWEN35_LOOKUP_MIN` (8), `IE_QWEN35_LOOKUP_ROWS` (8) | the repeated span that starts a lookup; the rows of an exact round |
| `IE_Q35MOE_LOOKUP`, `IE_Q35MOE_LOOKUP_ROWS` (128), `IE_Q35MOE_LOOKUP_MIN` (16) | 35B-A3B split at `--parallel 1` |
| `IE_Q35MOE_LANES_LOOKUP`, `IE_Q35MOE_LOOKUP_GAP` (2) | 35B-A3B split with request lanes: lookup rounds on the lanes; the steps every other decoding lane makes between two rounds |
| `IE_Q4E_LOOKUP`, `IE_Q4E_LOOKUP_MIN` (32), `IE_Q4E_LOOKUP_ROWS` (512) | Flash-Next at `--parallel 1` |
| `IE_DN_LADDER`, `IE_DN_LADDER_STEP` (8192) | in-place restart points on the 27B and 35B-A3B splits (one lane, prompt cache on): copies of the DeltaNet state while a prompt is read; the spacing doubles until 16 copies cover `--ctx` |

Diagnostics: `IE_Q35MOE_LOOKUP_STATS=1` / `IE_Q4E_LOOKUP_STATS=1` print a request's lookup counts; `/health` carries a
`lookup` object on the 35B-A3B lanes; a restart prints a `[dn-ladder]` line.

## Kernel switches (v0.2.6)

Each new kernel sits beside the one it replaces; `=0` selects the previous kernel.

| Switch | Kernel | Output |
|---|---|---|
| `IE_Q8_MOE_DOWN_V2`, `IE_Q8_MOE_GATEUP_V2`, `IE_Q8_MOE_DECODE_V2` | Q8_0 MoE prefill down, prefill gate+up, and decode / lane rows on the 35B-A3B class, with the native integer dot | bit-identical |
| `IE_Q8_SOA_BATCHED_V2`, `IE_Q8_SOA_GEMV_ND`, `IE_Q8_SOA_GEMV_G_ND` | Q8_0 GEMVs for 2–16 rows and for one token on Qwen3.8-27B, the 35B-A3B class and Qwen3.8-Flash, with the native integer dot | bit-identical |
| `IE_Q35_XMX_DECODE` | XMX decode attention at head size 256 from `IE_Q35_XMX_DECODE_MIN` (4096) tokens of context, 27B and 35B-A3B splits | not bit-identical; perplexity within 0.002 nats |
| `IE_FA2_VEC_PF`, `IE_FA2_TILE_PF`, `IE_FA2_COMBINE_SPLIT` | K/V prefetch (8 positions ahead; next tile) and the widened combine pass (16) in decode attention | bit-identical |
| `IE_Q35_ROWS_ATTN` | the lane rows' decode attention in one launch per pass | bit-identical |
| `IE_QWEN35_SPLIT_KQ` | native Q6_K / Q5_K on the 27B and 35B-A3B splits | `=0`: the previous fallbacks |
| `IE_Q35MOE_SHEXP_GATE_V0` | `=1` selects the previous shared-expert gate kernel (35B-A3B class) | bit-identical |
| `IE_DECODE_PROF` | `=1` prints a per-token decode breakdown on the 27B and 35B-A3B splits | diagnostic |

The first request after an upgrade compiles the new kernels once (measured: 4.87 s to the
first token on the 35B-A3B class, 6.95 s on Qwen3.8-Flash); the result is cached on disk.

## Historical memory observation, September 5

GGUF file size, mapped address space, process-resident RAM, Linux file cache and
GPU allocations measure different things. The streaming GLM backend maps the
shards and copies selected experts into its GPU caches. Mapping a file does not
make a separate private RAM copy of every byte. The OS can keep the underlying
pages in reclaimable file cache; many RAM monitors exclude that cache from the
headline used-memory number. Missing cached pages can be faulted in from disk.

During the September 5 follow-up, the owner's running server mapped all six
Q4_K_XL shards, totaling **199,707,321,347 bytes**, or 199.7 GB / 186.0 GiB.
A read-only `mincore` check found **191,230,578,688 bytes** of those file pages
resident in Linux's cache, about 191.2 GB / 178.1 GiB. The approximately
17/16 GiB GPU readings include the roughly 10 GiB expert cache on each card,
resident weights, context and workspaces. They do not indicate a smaller model
was substituted. These are point-in-time measurements, not pinned-memory
guarantees. The server does not use the optional MTP head in the mapped weights.

The revised reasoning build passed five host test targets, including exact GLM
prompt strings for all effort levels. An independent render of the actual GGUF
Jinja template matched all three levels byte for byte. Actual capability/editor
integration and final Dream focused tests passed; the full Dream suite passed
1816 tests with 35 skips before the final medium-alias regression was added.
The final focused run passed 108 tests. No new model was loaded and the owner's
existing server was left running. Evidence: `results/machx-reasoning-20260905/`.

## Dream workflow

Restart Dream, then choose **Model → GPUs → Context → Model tuning**. The table
shows only controls advertised for the selected architecture. Enter a setting's
number or name to edit it, `default` to reset it, Enter to start, or `cancel` to
leave. Stop strings use a JSON array such as `["END", "\nUser:"]`.

Settings apply to that launch and Dream session. They are not persisted as model
presets. Dream propagates them to both server arguments and HTTP generation
requests. `compact` overflow elides older history with recovery notes; `error`
refuses an oversized request without automatic history elision.

## Observed qualification, September 5, 2026

The local six-part GLM-5.3-Flash UD-Q4_K_XL loaded through Dream's real launcher
on two Arc Pro B70 cards at context 200000. `/props` reported 200000, and two
consecutive greedy short requests returned `Hello!`. This run used the default
CPU/GPU expert path with eight CPU threads. The server shut down cleanly.

At context 8192, live HTTP checks passed for repeated requests, SSE, literal
stops, reasoning, a synthetic tool-call/result round trip, invalid sampling
input and overflow errors. The greedy greeting matched the standalone runner
with special-token chat-marker encoding aligned and `IE_G5_CPU_MISS=0`; this
was a short reply comparison, not a broad logits or quality oracle.

Nine focused host test targets and the GPU sampling/count checks passed. Dream's
full suite passed with 1804 tests and 35 skips. A first full run hit an existing
intermittent browser-disconnect teardown failure; the repeated full run passed.
Logs and local evidence are in `results/machx-server-20260905/`.

Full 200000-token prompt quality, long-context throughput and every other model
were not qualified here. The shared architecture fix removes this loading
blocker; it does not establish universal model compatibility or a speedup.
