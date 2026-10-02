# `ie serve --config`: launch layouts and card pinning

Phases P1 (layouts, `--cards`) and P2 (`ie supervise`, card locks, served names) of the multi-model program
(`~/ds41_work/p60/multimodel_parallel_plan.md`). Everything here is additive: without `--config` and `--cards` the
engine parses its arguments and picks its GPUs exactly as before.

## Help

`ie --help`, `ie help <command>` and `ie <command> --help` (or `-h`) print the command's flags, defaults, architecture
restrictions and the environment variables that matter, and exit 0 without loading anything. `ie pull --help` is
still answered by the `ie-pull` helper. `tests/unit/cli_help_test.cpp` fails when the parser accepts a flag that the
run/serve help does not name: it reads the flag list from `include/ie/serve_options.hpp` itself.

## Cards: `--cards` and `ie cards`

A **card** is a discrete Level Zero GPU. Card N is the N-th one in PCI address order; the integrated GPU is never a
card. `ie cards` lists them without loading anything. On this machine:

```
$ ie cards
card 0  0000:04:00.0  Intel(R) Arc(TM) Pro B70 Graphics  (level_zero:0)
card 1  0000:09:00.0  Intel(R) Arc(TM) Pro B70 Graphics  (level_zero:1)
-       0000:00:02.0  Intel(R) Graphics  (level_zero:2, integrated: not a card)
```

These are the same numbers `xpu-smi discovery` prints as Device ID.

`--cards 1` (or `0,1`; the order does not matter) limits the process to those cards. Before the SYCL runtime starts, the
engine enumerates Level Zero itself, maps the cards to the runtime's Level Zero indices and sets
`ONEAPI_DEVICE_SELECTOR` to exactly those (`[ie] --cards: using card 1 (0000:09:00.0); ONEAPI_DEVICE_SELECTOR=level_zero:1`).
Every device list in the engine (the allocator, the VRAM planner, the DeepSeek and MiMo runtimes, preflight) then sees
only the chosen cards, so nothing is allocated on the others. Once the runtime is up, the engine checks by PCI address
that it exposes exactly the chosen cards and refuses to start otherwise.

Measured 2026-09-25, ~23:40 CDT (DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M, ctx 4096, `xpu-smi stats`, idle 26 / 34 MiB):
`--cards 1` put 2292-2363 MiB on card 1 while card 0 read 28 MiB; `--cards 0` put 2284-2355 MiB on card 0 while
card 1 read 36 MiB. The +2 MiB on the hidden card is also there with the old binary pinned by
`ONEAPI_DEVICE_SELECTOR=level_zero:1` (the `hermes-engine --pair` practice), so it does not come from `--cards`
(most likely the Level Zero driver's per-process state; not verified further). Two servers at once (`--cards 0` on :11461 and a layout with `"cards": [1]` on :11460) loaded 2286 /
2294 MiB and answered concurrent requests.

- The default (no `--cards`) is every card, as before.
- `--gpus` still says how many of the visible cards one model is split across (0 = automatic). `--gpus N` larger
  than the number of chosen cards is refused at parse time.
- `--cards` refuses to run with a restricting `ONEAPI_DEVICE_SELECTOR` (anything but `level_zero:gpu` /
  `level_zero:*`) or with `ZE_AFFINITY_MASK`, because those renumber the devices.
- `ie preflight --cards 1` plans against card 1 only; `ie bench --cards 1` passes the selection to `ie-bench`
  through the environment.
- Why the runtime selector and not a filter inside each device list: the engine enumerates GPUs in about ten places
  (four runtimes plus the planner), several of them in files that key the DeepSeek-V4.1 disk prompt cache
  (`src/model/deepseek41_numerics_inputs.txt`). One selector set before the runtime starts covers all of them, keeps
  those files byte for byte unchanged, and the hidden cards never appear as SYCL devices at all.

## Layout files

JSON, parsed with the vendored `third_party/nlohmann/json.hpp`. A layout is an object with a `servers` array (and an
optional `"version": 1`). Each server is an object of:

| key | meaning |
| --- | --- |
| `name` | the server's name (required in multi-server layouts). When a layout names the server, `/v1/models` reports this name as the model id, with `"root"` = the model file's id (see Served names) |
| `model` | the model path (`.gguf`, split shard 00001, or a V4.1 / MiMo checkpoint directory). `~/` expands to `$HOME`; a relative path is relative to the layout file |
| `cards` | array of card numbers, e.g. `[1]` or `[0, 1]` (= `--cards`) |
| `env` | object of environment variables, e.g. `{"IE_MIMO26_DFLASH": 7}`; strings, numbers, booleans (1/0) |
| `restart` | `ie supervise` only: `"none"` (default) or `"on-failure"` (see Supervisor); `ie serve` ignores it |
| every launch flag | spelled with `_`: `ctx`, `gpus`, `host`, `port`, `parallel`, `max_queue`, `slot_ctx`, `prefill_chunk`, `threads`, `vram_reserve_gib`, `int8_kv`, `spec`, `spec_k`, `spec_head`, `spec_draft`, `temp`, `top_k`, `top_p`, `min_p`, `repeat_penalty`, `repeat_last_n`, `presence_penalty`, `frequency_penalty`, `seed`, `max_tokens`, `stop` (array), `thinking` (true/false or "on"/"off"), `reasoning_effort`, and `prompt_cache` (true/false; false = `--no-prompt-cache`) |

`parallel` takes a number, 1 to 16. Since v0.2.6, a server without the key picks its own number of request lanes at
load, as `ie serve` does without `--parallel`; the string `"auto"` is not accepted in a layout.

Top-level keys besides `servers` and `version`, all for `ie supervise` (`ie serve` ignores them): `default` (the
server a request without `"model"` goes to; must name a server), `host` and `port` (the supervisor's front endpoint).

The meaning, range and default of each flag key are those of the flag (`ie serve --help`). The file is turned into
those flags and validated by the same parser as the command line, so a layout accepts exactly what the command line
accepts. Unknown keys, wrong types and out-of-range values are errors that name the entry and key
(`servers[0].ctx must be an integer`, `servers[0] (mimo): --ctx requires an integer in [9, 2147483647]`).

### Precedence

1. Flags on the command line override the layout: `ie serve --config mimo.json --ctx 65536`.
2. A `<model>` argument overrides the layout's `model`: `ie serve other.gguf --config mimo.json`.
3. `--stop` on the command line replaces the layout's `stop` list (instead of adding to it).
4. `int8_kv`, `spec` (true) and `prompt_cache` (false) have no negating flag: change them in the file.
5. An `env` variable already set in the process environment wins over the layout's value: the layout sets only the
   ones that are unset (`IE_MIMO26_DFLASH=0 ie serve --config mimo.json` turns DFlash off).

### (a) MiMo across both cards (today's default)

```json
{
  "version": 1,
  "servers": [
    {
      "name": "mimo",
      "model": "~/models/MiMo-V2.6-Flash-RL",
      "cards": [0, 1],
      "host": "127.0.0.1",
      "port": 11435,
      "ctx": 131072,
      "vram_reserve_gib": 1.5,
      "max_queue": 8
    }
  ]
}
```

`ie serve --config mimo.json` loads and runs the model exactly as `ie serve ~/models/MiMo-V2.6-Flash-RL --host
127.0.0.1 --port 11435 --ctx 131072 --vram-reserve-gib 1.5 --max-queue 8 --cards 0,1` does, with one visible
difference: because the layout NAMES the server, it reports the model as `mimo` -- `/v1/models` answers
`{"id": "mimo", ..., "root": "MiMo-V2.6-Flash-RL"}` and every reply and stream chunk carries `"model": "mimo"` -- where
the plain command reports `MiMo-V2.6-Flash-RL` (no `root`). Leave `name` out of a single-server layout to keep the
file's id. It also takes the card locks of cards 0 and 1 (see Card locks), which the plain command without `--cards`
does not. With `"cards": [0, 1]` on a machine that has exactly those two cards, the result is the same set of devices
as leaving `cards` out.

### (b) one model on card 0, another on card 1

As one multi-server layout, run by `ie supervise --config` (below). `ie serve --config` on it answers `this layout
lists 2 servers; run them all with ie supervise --config <layout>, or one with --server NAME`:

```json
{
  "version": 1,
  "servers": [
    {
      "name": "distill-qwen",
      "model": "~/models/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
      "cards": [0],
      "port": 11460,
      "ctx": 8192
    },
    {
      "name": "coder",
      "model": "~/models/other-model.gguf",
      "cards": [1],
      "port": 11461,
      "ctx": 16384,
      "env": {"IE_SERVE_NO_THINK": "1"}
    }
  ]
}
```

A multi-server layout must name every server and give each a `model`, a `port` and `cards`; names and ports must be
unique (the same port is a clash when the RESOLVED bind addresses overlap: `localhost` = `127.0.0.1`, and a wildcard
`0.0.0.0` or `::` clashes with every address) and no card may be listed by two servers.

`ie serve --config two.json --server coder` runs just that entry. Without a layout, the same thing runs as two
processes with plain flags:

```
ie serve ~/models/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf --cards 0 --port 11460
ie serve ~/models/other-model.gguf --cards 1 --port 11461 --ctx 16384
```

## Supervisor: `ie supervise --config <layout.json>`

One process that starts every server of a layout as its own `ie serve` child and serves ONE OpenAI-compatible endpoint
in front of them, routing each request by its `"model"` field. The front is a router only: the children do all the
work, each with its own model, cards, port, context, admission queue and prompt cache.

```json
{
  "version": 1,
  "default": "distill-card0",
  "port": 11470,
  "servers": [
    {"name": "distill-card0", "model": "~/models/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
     "cards": [0], "port": 11471, "ctx": 4096},
    {"name": "distill-card1", "model": "~/models/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
     "cards": [1], "port": 11472, "ctx": 4096, "restart": "on-failure"}
  ]
}
```

```
ie supervise --config two.json                 # front port: --port, else the layout's "port" (11470), else 11435
curl -s localhost:11470/v1/models
curl -s localhost:11470/v1/chat/completions -d '{"model":"distill-card1","messages":[{"role":"user","content":"hi"}]}'
curl -s -X POST localhost:11470/admin/shutdown
```

`ie supervise --host H --port P` override the layout's top-level `host` / `port`.

**Children.** Each server runs as `ie serve --config <layout> --server <name>` (the same executable), so it is the
exact P1 single-server launch: the entry's flags, its `env` (a variable already set in the supervisor's environment
wins), `--cards` pinning, its port. Every server needs `name`, `model`, `cards` and `port`, and its `host` must be
reachable over loopback (`127.0.0.1`, `localhost`, `::1`, or a wildcard `0.0.0.0` / `::`), because the supervisor
stops it through its loopback-only `/admin/shutdown`. The front port may not clash with a child's. Children run in
their own process groups, so a terminal ^C reaches only the supervisor. Each child's stdout and stderr go through a
pipe to the supervisor, which writes every line to its own stderr prefixed with the server name
(`[distill-card1] [ie] serving distill-card1 on ...`); the supervisor's own lines start with `[ie supervise]`.
stdout and stderr use separate pipes, so a partial stdout line is never joined to a stderr line (lines from the two
streams may still interleave in a different order than the child wrote them). A child
runs with SIGPIPE ignored, so if the supervisor dies the child's log writes fail instead of killing it.

**Readiness.** A child is `loading` until its `/health` answers 200 (`ie serve` starts its listener after the load),
then `ready`. The supervisor polls every child's `/health` once a second. A 503 from a child is `unhealthy` (a lost
device: it needs a restart) or `stopping`.

**Routing.** `POST /v1/chat/completions` and `POST /v1/completions` go to the server whose `name` equals the body's
`"model"`; failing that, to the one server whose model id (the model file name without `.gguf`, or the checkpoint
directory's name) equals it. The request body is forwarded unchanged and the child's reply comes back unchanged:
status, headers, and the body byte for byte, a Server-Sent-Events stream included (every event and field as the
child wrote it -- `timings`, `ie_vitals`, `truncated_tool_call`, comments -- written through as each piece arrives).
A client that disconnects is disconnected upstream too, while streaming and while the child has not answered yet, so
the child aborts the generation as it would for a direct client.

| request | answer |
| --- | --- |
| `"model"` = a name, or a model id only one server has | that server's reply |
| no `"model"` (or `null`) | the layout's `default` server; `400 model_required` naming the models when there is none |
| a model id two servers share | `400 model_ambiguous` naming them |
| anything else | `404 model_not_found`: `model "x" not found; available models: a, b` |
| the server is still loading | `503 loading` (`Retry-After: 5`) |
| the server has exited | `503 server_not_running` with its exit code or signal |
| the supervisor is stopping | `503 shutting_down` |
| the body is not a JSON object | `400 invalid_json` |

`ie serve` has no `/v1/completions` today, so that path is forwarded and the child answers 404.

**Other endpoints.**

- `GET /v1/models`: every server, `{"id": <name>, "object": "model", "owned_by": "local", "root": <model id>,
  "status": "loading|ready|unhealthy|stopping|exited|stopped|left_running"}`.
- `GET /health`: `{"status", "inflight", "queued", "default", "servers": {<name>: {"state", "model", "port",
  "cards", "pid", "restarts", "health": <the child's own /health>, "exit_code" | "exit_signal", "loading_s"}}}`.
  `status` is `ok` (every server ready, HTTP 200), `degraded` (at least one ready, 200), `loading` (none ready yet,
  503), `unavailable` (503) or `stopping` (503). `inflight` / `queued` are the children's sums, as of the last poll
  (at most one second old).
- `GET /props?model=<name or id>`: that server's `/props`, unchanged; without `model`, the default server's (400
  without a default).
- `POST /admin/shutdown` (loopback callers only; 403 otherwise): the same as SIGTERM.

**A child that exits** is reported (`server "x" (pid N) exited with code C after S s -- it is down; the other servers
keep serving`, or `was killed by signal S`, and exit 75 is named `device lost`), `/health` shows it, and requests for it
get 503; the other servers keep serving. With `"restart": "on-failure"` a child that exits non-zero or by a signal is
started again, at most 3 times, never after exit 75 (a lost device needs a person) and never while stopping. The
default is `"restart": "none"`.

**Shutdown** (SIGTERM, SIGINT or `POST /admin/shutdown`): new requests get 503; then every child, in parallel, gets
its own `POST /admin/shutdown` (which aborts its in-flight generation at the next token, as `SERVER_CONTROLS.md`
describes) and the supervisor waits up to 600 s for it to exit. A child that is still LOADING is not interrupted: the
supervisor waits up to 1800 s for it to become ready and then stops it the same way. The supervisor never sends a
child a signal and never kills it: killing a process that holds a model can wedge the GPU. A child that does not stop
within the time is reported (`LEFT RUNNING (not killed) -- check it, then POST http://127.0.0.1:<port>/admin/shutdown`)
and left alone. The front endpoint closes after the children. A second SIGINT/SIGTERM exits the supervisor at once and
leaves the children running (they are in their own process groups; stop each through its own `/admin/shutdown`). A
supervisor killed with SIGKILL leaves its children running the same way, still holding their card locks. Exit status:
0 = every child's process has exited with code 0; 1 = a child is still running (`stopped; STILL RUNNING (not killed):
...`) or a child's last exit was a signal or a non-zero code -- during its requested shutdown included (a crash or exit
75 in teardown is not a clean stop) -- named in `stopped; N child(ren) did not stop cleanly: name (signal 6), ...`;
2 = nothing was started (a card lock is held, the front port could not be bound, the layout is invalid). A child that
crashed and was restarted counts by its last run.

## Card locks

At most one engine process per physical card. Card N's lock is an exclusive `flock(2)` on `/tmp/ie-card-N.lock`
(`IE_CARD_LOCK_DIR` moves the directory; the tests use their own). The kernel drops a lock when the holder's last
descriptor closes, so a crashed process never leaves a stale lock; the file only carries the holder's pid for the
message.

| who | locks it takes |
| --- | --- |
| `ie serve` / `ie run` with `--cards LIST` (or a layout `cards`) | LIST's card locks, at startup; refuses to start (exit 2) when one is held: `card 0 is in use by another engine process (pid N) -- lock /tmp/ie-card-0.lock` |
| `ie serve` / `ie run` without `--cards` | none (unchanged) |
| `ie supervise` | every child's cards, before it starts any child (a held card: exit 2, nothing started); each child inherits its own cards' lock descriptors, so a lock lives as long as the supervisor or that child |
| `scripts/ie-run-guarded --cards LIST` | LIST's card locks only: guarded runs on different cards coexist, two on the same card are refused (exit 3) |
| `scripts/ie-run-guarded` without `--cards` | the single-flight lock `/tmp/ie-gpu.lock` as before, plus card locks 0..7: a whole-machine run and any per-card run exclude each other |

A process started under a lock holder is told which cards are already held for it through `IE_GPU_LOCK_HELD` (`all`
or a list like `0,1`; `ie-run-guarded` and `ie supervise` set it) and does not take those again: a second lock on the
same file from a new descriptor would conflict with the inherited one. So `ie-run-guarded --cards 0,1 ie supervise
--config two.json` and `ie-run-guarded ie serve m.gguf --cards 1` both work. The locks do not cover engine binaries
that do not know them (`ie-mimo26-*`, `ie-bench` and other research tools run directly); `ie-run-guarded` without
`--cards` still keeps those single-flight. Two per-card `ie-run-guarded` runs each get their own `--mem` cap: size
them for the machine together.

## Served names

When a layout names the server (`"name"`), `ie serve --config` reports that name as the model id: in `/v1/models`
(`{"id": "coder", ..., "root": "Qwen3-Coder-30B-A3B-Q4_K_M"}`, `root` = the model file's id) and in the `"model"` field
of every reply and stream chunk. `ie serve <model>` without a layout, or a layout without `name`, reports the model file's
id exactly as before, without `root`.

## What Dream needs for a supervised layout (P3)

- One base URL (the front) for every model. Dream's model choice for a role becomes the `"model"` field of each request
  = the layout's server name; `GET /v1/models` lists them (`root` = the file, `status` = whether it can answer now).
- Per-model context and capabilities: `GET /props?model=<name>` (the child's own `/props`, e.g. `n_ctx`). A bare
  `/props` answers for the `default` server only, so Dream should always pass `model` when a layout has several.
- Health: the front's `/health` has per-server `state`; Dream should treat a server that is `exited` / `unhealthy` as
  unavailable for that role instead of treating the whole endpoint as down (the front answers 200 while any server is
  ready).
- Launcher: start `ie supervise --config <layout.json>` (under `ie-run-guarded --cards <all cards the layout uses>` if
  it wants the memory cap) instead of `ie serve <model> ...`, wait for `/health` `status: ok` (or for the servers the
  session needs to be `ready`), and stop it with `POST /admin/shutdown` on the front, never a kill: shutdown can take as
  long as the slowest child's load plus its stop. `ie serve --config <layout> --server <name>` runs one entry alone.
- Streaming is unchanged (Dream's SSE parser sees the child's bytes); the reply's `"model"` is the server name.
- Per-card placement is the layout's business: one server per card set, no card in two servers (the parser refuses it).

## Measured 2026-09-26 00:07 (GPU)

Layout: the one above (the 1.5B distill on card 0 and on card 1), front on 11470, launched as
`scripts/ie-run-guarded --cards 0,1 --mem 48G --timeout 480 build/src/ie supervise --config two.json`.
**Card 0 was already in a GT reset loop** (kernel: `Engine memory CAT error [18]: class=bcs` at 00:01:43, then
repeated `Kernel-submitted job timed out` / GT resets) when this ran, so the card-0 child failed to load
(`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, SIGABRT after 3 s). This run therefore shows the one-child-down case; the
two-servers-on-two-cards check was run separately by the P2 gate after the reboot and passed (its report holds those
numbers; they are not repeated here). What this run observed:

- the supervisor reported the dead child (`server "distill-card0" (pid 889692) was killed by signal 6 after 3 s -- it
  is down; the other servers keep serving`), `/health` said `degraded` with `distill-card0: exited, exit_signal 6`,
  requests for it got `503 server_not_running`, and `distill-card1` kept serving;
- card 1 loaded in 3 s (2292 MiB, 2363 MiB while generating) while card 0 read 28 MiB; routing by name worked, an
  unknown model got the 404 listing both names, the shared model id got `400 model_ambiguous`; the child's
  `/v1/models` said `{"id":"distill-card1",...,"root":"DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"}`;
  `/props?model=distill-card1` returned the child's `/props` (`n_ctx` 4096);
- a greedy 226-token reply took 1.38 s through the front;
- streaming through the front vs the same request straight to the child: 48 events each, 8400 bytes each, the same
  keys in every event and the same text (`Content-Type: text/event-stream` both); `ie_vitals` did not appear on this
  architecture, so the non-standard fields are covered by the unit test's byte comparison only;
- `POST /admin/shutdown` on the front: the child got its own `/admin/shutdown` and exited 0 in about 6 s; supervisor
  exit 1 (because the card-0 child had failed); no engine process left; after 30 s the cards read 26 / 34 MiB.


## Tests

- `serve_config_test` (host only): layouts, types, ranges, precedence, `env`, the multi-server shape, `--cards`
  parsing and the card numbering; every `json` example in this document must parse.
- `supervisor_test` (host only): routing (name, unique id, ambiguous id, unknown, missing, default), `/health`
  aggregation and `/v1/models`, the card locks (same card refused across processes, other cards fine, released on
  close, `IE_GPU_LOCK_HELD`), and the supervisor against fake `ie serve` children: child output prefixed with the server name, byte-exact SSE passthrough of two
  concurrent streams, a child's 400 passed through, client disconnects reaching the child (mid-stream and before the
  first byte), `/props`, a crashed child reported while the others serve, ordered `/admin/shutdown` with no signal, a
  loading child stopped only after it finished loading, a child that ignores shutdown left running (exit 1 from its live pid, whatever its state was polled as), on-failure
  restarts (limit; none after exit 75), refusal on a held card.
- `cli_help_test` (host only): the help names every parser flag, every parser flag has a layout key, every command
  `main.cpp` dispatches has help.
