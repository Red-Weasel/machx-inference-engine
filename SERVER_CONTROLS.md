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
| `--parallel`, `--slot-ctx` | Request slots and per-slot context; architecture restrictions apply |
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

## Admission, health and shutdown

`ie serve` runs up to `--parallel N` generations at once (default 1; only the
Qwen3.6-27B two-card split actually batches, every other architecture takes
whole-generation FIFO turns). At most `--max-queue M` further requests wait for
a slot (default 8); beyond that the server answers **HTTP 429** immediately
with `Retry-After: 1` and `{"error":{"code":"queue_full"}}`, so a burst can
never exhaust the HTTP worker pool. The pool is sized `parallel + max_queue + 4`
so `/health`, `/props` and the 429s themselves are always answered.

`GET /health` returns `{"status":"ok","inflight":..,"queued":..,"parallel":..,"max_queue":..}`
with 200 while the device is healthy. If a forward ever reports a lost or reset
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

Image inputs are refused with an error when the server runs `--parallel > 1`:
per-request image staging is engine-global and not slot-safe.

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
