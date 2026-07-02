# Seal-on-Engine: First Integration Record (P1 Task 7)

Date: 2026-06-10
Engine: `ie serve` (this repo, main @ 9ec2d94), model `Qwen3.6-35B-A3B-Q4_K_M.gguf`, port 11435, ctx 8192
Seal: `/home/weezy/SEAL - VERSION 2.1` branch `snapshot/2026-06-10-pre-stabilization`, `engine/` (Rust, seal-engine v0.2.0)
Verdict (original run): plumbing works end-to-end (connect, stream, parse); agent tasks BLOCKED because the engine ignored the OpenAI `tools` field.
**Update P1.7b (same day): tools gap fixed — Seal agent tasks now PASS end-to-end. See §P1.7b below.**

## Working env/config recipe (env vars only — no Seal file changes needed)

Seal resolves its provider from env in `engine/src/providers/mod.rs::ProviderConfig::from_env`:

```bash
# start the server
./build/src/ie serve /home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf --port 11435 --ctx 8192

# run Seal headless against it
SEAL_RUNTIME=openai-compat \
SEAL_RUNTIME_URL=http://127.0.0.1:11435 \
SEAL_MODEL=Qwen3.6-35B-A3B-Q4_K_M \
OLLAMA_CONTEXT_LENGTH=8192 \
"/home/weezy/SEAL - VERSION 2.1/engine/target/release/seal-engine" \
  --permission-mode danger-full-access --one-shot --tool-call-cap 12 \
  prompt "<task text>"
```

Notes on resolution order (verified in source):
- Base URL: `SEAL_RUNTIME_URL` > `ANTHROPIC_BASE_URL` > `OLLAMA_BASE_URL` > default. The
  openai-compat default is `http://127.0.0.1:8080/v1` (the audit's `11434` default only applies
  to the ollama runtime). Seal auto-appends `/v1` if missing, so bare `http://127.0.0.1:11435` is fine.
- Model: `SEAL_MODEL` > `ANTHROPIC_MODEL` > `OLLAMA_MODEL` > `"local-model"`. Must be set —
  Seal does NOT auto-discover the chat model from `/v1/models` (it only uses `/v1/models` for
  `ping()` health checks and embed-model name discovery).
- API key: `SEAL_API_KEY` / `OPENAI_API_KEY`, optional — not required, our server ignores auth. Not set.
- Context budget: `OLLAMA_CONTEXT_LENGTH` (yes, that name even for openai-compat), default 32768;
  set to 8192 to match `--ctx`.
- Optional sampling knobs: `SEAL_TEMPERATURE` (default 0.2), `SEAL_TOP_P` (default 1.0).

Headless entry point: `seal-engine [flags] prompt <text>` (hand-rolled CLI in `engine/src/main.rs`,
no clap). `--one-shot` exits after the first task; `--permission-mode danger-full-access` avoids
stdin approval prompts. The eval harness (`scripts/supercharger-eval.ps1` → `eval-results/`) is
PowerShell/Windows-only, so the one-shot prompt path is the cheapest Linux E2E run.

## What ran and what happened

Build: `cargo build --release` on the snapshot branch — clean (18 warnings, 0 errors).

Two one-shot agent tasks in a disposable workspace (`create hello.txt, read it back, verify`):

| Aspect | Result |
|---|---|
| Connect / ping (`GET /v1/models`) | OK |
| Chat request accepted (messages, temperature, top_p, stream, tools, tool_choice) | OK — no 400s |
| SSE streaming | OK — deltas flowed token-by-token through Seal's `[seal::delta]` sink; latency felt instant, smooth decode throughout |
| Context-length finishes | None observed in agent runs (`finish_reason":"length"` only when forced via tiny `max_tokens` probe) |
| Structured `tool_calls` in response | **Never** — engine has no tool support |
| Seal's text-fallback tool-call parser | **Did not fire** — it recognizes JSON `{"name":...,"arguments":...}` shapes; the model emitted XML instead |
| Task completion | **FAILED both runs** — agent ended after step 1 with zero tool calls; `hello.txt` never created |

What the model actually emitted (because it never saw tool definitions): improvised XML pseudo-calls,
run 1: `<write_file path="hello.txt" content="hello from seal\n">`, run 2: Anthropic-style
`<tool_calls><invoke name="write_file"><parameter name="path">...` — neither matches Seal's
structured channel nor its JSON fallback parser. Seal's own system prompt explicitly tells the model
NOT to emit text-format tool calls ("use the structured tool_calls field"), so this won't fix itself
with prompting.

Seal-side resilience observed: the missing `/v1/embeddings` (404) is handled gracefully — memory
auto-retrieval is skipped with a one-line warning (4xx is non-retried), the agent continues.

## Engine-side gaps (in priority order)

Statuses as of P1.7b: gap 1 **FIXED** (template-side; structured `tool_calls` extraction
deferred — Seal's text fallback covers it), gap 2 **FIXED** (501), gap 3 **FIXED**,
gap 4 open (single-flight), gap 5 **FIXED** (null/absent content accepted on assistant
tool_calls messages).

1. **`tools` / `tool_choice` silently ignored** (`src/server/openai_proto.cpp` parses only
   model/messages/temperature/top_p/max_tokens/stream/enable_thinking). The tool definitions are
   never rendered into the Qwen chat template (`<tools>...</tools>` system block), so the model
   doesn't know the Qwen-native `<tool_call>{json}</tool_call>` emission format. **This is the
   single blocker for Seal agent tasks.** Fix = (a) render `tools` into the chat template, and
   (b) parse `<tool_call>` spans out of generated text into OpenAI `tool_calls`
   (message field + streamed `delta.tool_calls` with index/function.name/function.arguments).
   Note: even just (a) might suffice for Seal — Qwen's native emission format
   (`<tool_call>{"name":...,"arguments":...}</tool_call>`) is exactly what Seal's fallback
   parser recovers from plain content text.
2. **No `/v1/embeddings`** (404). Seal degrades gracefully (memory auto-retrieval off), so this is
   nice-to-have. If added, prefer returning 501 when unsupported — Seal special-cases 501 as
   permanent and stops retrying/spamming.
3. **`role:"tool"` / assistant `tool_calls` history messages**: accepted without error (engine
   requires only `role` + `content` per message), but the `tool_calls`/`tool_call_id` fields are
   dropped on the floor — once gap 1 is fixed, multi-turn tool conversations need these rendered
   into the template properly (Qwen `<tool_response>` blocks).
4. Single-flight mutex (`gen_mu`) serializes generations — fine for one Seal agent, will bite with
   Seal subagents fan-out later.
5. Engine requires a `content` key on every message — strict-OpenAI clients sometimes omit
   `content` on assistant tool_calls messages (Seal always sends it, so not blocking today).

What already works well: /health, /v1/models, non-stream + SSE chat with correct
`finish_reason` (`stop`/`length`) and `[DONE]` terminator, usage accounting, multi-message
histories, Seal's retry/backoff and base-URL `/v1` normalization all interoperate cleanly.

## Repo changes

- Engine repo: this file only.
- Seal repo: **no changes** — env-var-only integration, nothing to commit there.

## P1.7b — tools gap implemented and re-verified (2026-06-10)

Spec basis: docs/superpowers/specs/2026-06-10-gold-standard-roadmap-design.md §6.2 —
v1 passes tool definitions into the prompt via template and returns text; Seal's
tool-call-recovery fallback handles text-embedded tool calls.

### What was implemented

1. **`tools` rendered into the chat template** (gap 1a). `parse_chat_request` captures the
   raw `tools` array (`ChatRequest::tools_json`; non-array → 400 "tools must be an array");
   `build_chatml_prompt` gained a `tools_json` parameter (empty → byte-identical legacy
   output, goldens unchanged) that injects the **canonical Qwen3 JSON tools preamble** into
   the system turn (creating one if absent): `# Tools ... <tools>{schema per line}</tools>
   ... return a json object ... <tool_call>\n{"name": <fn>, "arguments": <args>}\n</tool_call>`.
   Convention note: research/04 §4.2 documents a Qwen3.6-native XML form
   (`<function=NAME><parameter=KEY>`), but the JSON convention was chosen deliberately —
   it is what Seal's text fallback parses, and the instruction-following preamble drives
   emission (verified below). Engine API: `Engine::chat(..., std::string_view tools_json = {})`;
   server threads `cr.tools_json` through stream + non-stream paths.
2. **Tool-turn history rendering** (gap 3). Assistant messages with `tool_calls` (content
   null/absent OK — gap 5) are converted to `<tool_call>\n{"name":..., "arguments":...}\n</tool_call>`
   text blocks (OpenAI's JSON-escaped-string `arguments` parse-and-redumped, embedded as raw
   JSON); `role:"tool"` messages render as `<tool_response>\n...\n</tool_response>` inside a
   user turn, consecutive tool turns merged into one user turn (official Qwen3 template rule).
3. **`/v1/embeddings` → 501** (gap 2) with `{"error":{"message":"embeddings not implemented"}}`
   so Seal permanent-skips instead of retrying a 404.
4. **Decode keep-list**: `Engine::generate` decodes with skip_special=true, which was
   silently stripping the model's emitted `<tool_call>`/`</tool_call>` special tokens
   (ids 248058/248059) from response text. `Tokenizer::decode` gained a `keep_special`
   span; the engine preserves exactly those two markers so clients can recover
   text-embedded tool calls. (Without this, gap-1a alone would have shipped bare JSON
   with no markers.)

Unit coverage: `tests/unit/openai_proto_test.cpp` (tools captured + key order preserved,
empty/non-array tools, tool_calls→text round-trip incl. unparseable-arguments and missing
content, tool message accepted, malformed tool_calls → 400) and
`tests/unit/tokenizer_test.cpp` (exact goldens: system+user+tools; 5-turn
user→assistant(tool_call)→tool→assistant→user; consecutive-tool merge; empty-tools
byte-identity). Full ctest: 12/12 green.

### Re-run of the exact one-shot recipe (this file, §recipe)

Task: `create hello.txt with the content: engine-test, then verify it exists`, cwd
`/tmp/seal_accept`, same env (SEAL_RUNTIME=openai-compat, port 11435, ctx 8192), Seal
snapshot branch untouched.

| Aspect | Result |
|---|---|
| Model emission | exact Qwen JSON form: `<tool_call>\n{"name": "write_file", "arguments": {"path": ..., "content": "engine-test"}}\n</tool_call>` |
| Seal text-fallback parser | **fired both times** — recovered both calls from streamed content |
| Tool calls executed | **2** — `write_file` then `read_file` (verification) |
| File | `/tmp/seal_accept/hello.txt` created, content exactly `engine-test` (11 bytes) |
| Outcome | **PASS** — agent completed all steps, exit 0, correct final synthesis ("Verified by reading it back.") |

Still deferred (not needed for Seal): structured `tool_calls` extraction into the OpenAI
response/`delta.tool_calls` (gap 1b), `tool_choice` enforcement, gap 4 (single-flight mutex).
