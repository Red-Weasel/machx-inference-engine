# V4.1 port — Phase 12 results: usable end to end

**Criteria:** `31_PHASE12_END_TO_END_CRITERIA_2026-09-13.md`. **Commits:** fe7eb8c (tokenizer),
de52921 (prompt format), 030f60a (generate loop + CLI), then the server route (below).

## Step 1 — the tokenizer (`ie-ds41-tokenizer-test`, criterion 1: PASS)

`Tokenizer::load_from_hf_json` reads the checkpoint's `tokenizer.json` (byte-level BPE, vocab
128,000 + 1,280 added tokens = 129,280, 127,741 merges) into the engine's existing BPE with the
DeepSeek-V3/V4 split the GGUF path calls "joyai-llm". Two things had to be right and were found
by the parity test, not by reading: (1) the chat-role tokens `<｜User｜>`, `<｜Assistant｜>`,
`<｜System｜>` are added tokens flagged `special: false` in the JSON, and HF `tokenizers` still
matches added tokens atomically — so every added token is a match literal and the flag decides
only skip-special on decode; (2) U+007E TILDE is a Unicode symbol (Sm), so under the JSON's
` ?[\p{P}\p{S}]+` split " ~" is one token — the engine's cascade already had a tilde-as-symbol
variant (`hyv4`), and the JSON loader selects it. Parity, token for token: the 300 KB docs corpus
(97,639 ids), a stress mix (digits, CJK, code, punctuation, emoji, case runs, whitespace, five
scripts, the role tokens: 487 ids), the pp2048 text round-tripped (2,048 ids); decode of the
reference's ids byte-exact on all three; decode(encode(x)) == x. The loader lives in its own
translation unit because the unit tests compile `tokenizer.cpp` directly and icpx crashed on the
JSON header under their flags; those tests are unchanged.

## Step 2 — the prompt format (`ie-ds41-prompt-test`, criterion 2: PASS)

`ds41_encode_messages` ports `encode_messages` of the checkpoint's `encoding/encoding.py` for
text turns: thinking and chat modes, the numeric reasoning-effort prefix (index 0 of a thinking
conversation; aliases low 50 / high 75 / max 100), mid-conversation `<｜System｜>` turns (which
count as user turns for the generation header), the drop-thinking rule, the transition tokens
only ahead of an assistant turn, `add_bos`. Fifteen cases through the reference's own Python
(single and multi-turn, both modes, three effort values, drop-thinking on and off, a mid
system in the middle and last, an assistant-last prefill, an empty system, no bos, non-ASCII):
every prompt equal byte for byte, and tokenising to the reference's ids. Tool turns, tool calls,
images, reminders and tasks are refused with a message.

**A criterion deviation, recorded (gate 12 finding 1):** criterion 2 named "the reference's
`encoding/tests` text cases", assuming the checkpoint shipped text cases; it ships five files,
four of which need what this port refuses (tool definitions, a `latest_reminder` turn with a
`task`, image content). The fifteen cases above are self-authored and run through the
reference's Python — the same oracle, but chosen by the implementer. After the gate the five
shipped files were put to work in the test: **file 2 is text-only and the port matches its
`test_output_2.txt` byte for byte (294 bytes; the runner's default for these files is chat
mode)**, a case chosen by someone else; files 1, 3, 4, 5 are refused for the stated reason
(tool definitions, reminder, reminder + task, image content). Putting them to work found a
fake: an image turn's content is a list the port cannot carry, and the port had rendered the
conversation without it — `Ds41ChatMessage::has_images` / `has_tool_calls` and
`Ds41PromptOptions::has_tools` now make the port refuse at its own layer (the engine sets them
from the request), instead of relying on the caller.

## Step 3 — the generate loop (`ie-ds41-generate-test`, criterion 3: PASS)

Host-side sampling over the fp32 logits (the engine's kernels take fp16 device logits; the
semantics — temperature, top-k, top-p, min-p, repetition penalty over a recent window — are
reproduced on the host, which costs nothing against a 190 ms step), streaming detokenisation
that holds back a partial UTF-8 sequence, stops on eos / budget / stop strings / callback, and
odd-length prompts prefilled even with the last token stepped (a prefill's length must be a
multiple of the compress ratio). Results: temperature 0 under top-k/top-p is the argmax; a seeded
draw repeats; top-k 1 is the argmax at any temperature; the penalty moves a recent argmax; greedy
from the golden prompt gives the golden's 17575 16 455 6102 (" Berlin. The capital"); an
11-token prompt runs; the stop string "." ends the run at " Berlin" unemitted; after the
2048-token pp text with replay ON the first four tokens are the replay test's 369 2619 1683 16,
at **192.0 ms/token** over 36 tokens on the step-3 build and 173.6 on the final build (the decode
test: 194.9 / 199.0 on the same two builds). Stated as what it is (gate 12 pre-read, P1): the
loop is faster than the decode test's bench by more than the 0.6 ms same-binary spread gate 11
measured on the decode test, so it is a real difference between the two harnesses, not noise;
a faster loop is not a regression, and the cause was not investigated. The generate test's own
run-to-run spread on two builds is 18 ms; no claim rests on it.

The property the odd-length rule rests on, measured directly (P3, then gate 12 finding 3 at
four lengths): prefill(T) against prefill(T-2) followed by two decode steps, both legal lengths,
the logits at position T-1 — see the regression section for the numbers.

## Step 4 — the CLI (`ie-ds41-run`, criterion 4), three sessions, all greedy

Load: 39 s on two cards (capacity 4,096 tokens; card 0 layers 0-19 with 45 static + 8 stream
slots per layer, card 1 layers 20-39 with 44 + 8; 228 pinned per layer), the engram tables and
the held-out ranking read from files beside the model.

**A question (thinking mode, effort 75):** `--prompt "What is the capital of Germany, and what
river runs through it? Answer in two sentences."` → 49 prompt tokens, prefill 3.21 s, 99
generated in 41.8 s = **2.37 tok/s**, stop: eos.
> We need answer. Need comply two sentences. Capital Berlin. River runs through Berlin: Spree,
> also Havel? […] "The capital of Germany is Berlin. The Spree River runs through Berlin." That's
> two sentences. Ensure no extra.\</think>The capital of Germany is Berlin. The Spree River runs
> through Berlin.

**A 2,000-token document with a question (thinking mode):** the pp2048 text (the V4.1 model
card) pasted with "In three sentences, what is this document about, and what is the single most
important number it reports?" → 2,107 prompt tokens, prefill 10.13 s = **208 tok/s**, 300
generated in 79.9 s = **3.76 tok/s**, stop: length (the reasoning used the budget; the answer's
first two sentences landed). The reasoning identified the document (DeepSeek-V4.1-Flash, 552B
parameters, 1M context, CED, CSA2, 45T pre-training tokens) and chose the 890 bytes per token
of global KV cache as the number.

**A three-turn chat (chat mode, `--chat` on stdin):** "My name is Matt and I build inference
engines for Intel Arc GPUs." / "What did I just tell you my name was, and what do I work on?" /
"Suggest one concrete optimization for MoE decode on such a machine, in two sentences." →
> Nice to meet you, Matt — that's a pretty specialized corner of the stack. […]
> Your name is Matt, and you build inference engines for Intel Arc GPUs.
> On Arc, keep the MoE expert weights resident in XMX-friendly tiled layouts and batch decode
> tokens across experts so each expert's GEMM hits the systolic array with high utilization
> rather than being launched as tiny per-token matvecs. […]

Each turn re-renders the whole history and prefills it again (no prompt-cache reuse across
turns yet: the third turn's prefill carries the first two turns). Ctrl-C ends the current answer
and the runtime unloads through the same path the tests check.

The decode rates are the Phase 11 numbers seen from the outside: ~2.4 tok/s at a 50-token
context (the short-prompt regime: 38-43% decode hit rate, 70+ mmap experts per token) and
3.8-5.2 tok/s once the context is long enough for the stream slots' locality to bite.

## Step 5 — the OpenAI route (criterion 5)

Chosen: the `Ds4Bundle` pattern inside `Engine` — a model DIRECTORY whose `config.json` names
`deepseek_v41` loads through `Engine::ds41_load` (src/engine/ds41_engine.cpp: the model, the
engram tables and the held-out ranking from files beside it, the tokenizer.json tokenizer, one
in-order queue per Arc card, the resident runtime with `--ctx` as its position capacity) and
never touches the GGUF machinery; `chat()`, `generate()` and `reasoning_effort_error()` branch
to it; the server's think-split gate, the capabilities table and the reasoning capabilities
learn `ModelArch::kDeepSeek41`. So `ie serve <model dir>` is the whole change for the user, and
the dedicated-server alternative was not needed. Tool definitions and images come back as
`finish_reason: "error: …"`, not as fake output; the V4 DSML tool-call parser stays off for
this arch.

`ie serve ~/models/DeepSeek-V4.1-Flash --ctx 4096 --port 8089`: `/health` ok after 42 s
(parallel 1, max-queue 8). `POST /v1/chat/completions` with the CLI's question at temperature 0
→ finish "stop", usage 49 prompt / 99 completion, `reasoning_content` the same reasoning the CLI
printed and `content` **"The capital of Germany is Berlin. The Spree River runs through
Berlin."** — the CLI's answer, byte for byte (criterion 5's CLI == server at temperature 0).
Streaming with `enable_thinking: false` ("Name three primary colors, comma separated, nothing
else.") → delta chunks "Red, B" / "lue, Yellow", a finish chunk, a usage chunk, `[DONE]`.
`POST /admin/shutdown` from loopback → "stopping", the process exited and unloaded. The
server-layer unit tests (openai_proto, server_capabilities, server_controls, server_admission,
glm5_server) PASS with the new arch in the tables.

## Regression (criterion 6) and refusals (criterion 7)

On the end-to-end build (`scratchpad/p11/chain6.log`, `p12_md5.txt`; one GPU job at a time):
tokenizer PASS, prompt PASS, tier PASS (8- and 3-slice byte identity), decode PASS (every
Phase 9 criterion under its bar, digits as the Phase 11 shipped build's: forced worst 9.36e-3 /
1.49e-2; 199.0 ms/token at the 2048-token context, 5.02 tok/s, hit rate 61.6%), generate PASS
(173.6 ms/token on this run; with the continuity check on the rebuild: PASS at 178.1 ms/token),
forward own PASS, forward forced PASS, resident PASS (pp512 109.4 / pp2048 280.1 cold, unload
within the residue bars), replay PASS (24/24 tokens, 24/24 closer, mean KL 0.0052). The GGUF
path is untouched: the V4.1 branch is an early return on a model directory, and the server-layer
unit tests (which load no model) PASS.

**Prefill-vs-decode continuity (P3, four lengths after gate 12 finding 3):** prefill(T)
against prefill(T-2) + two decode steps, the logits at position T-1, replay off:

| T | max relative difference | argmax |
|---|---|---|
| 12 (golden prompt) | 4.54e-2 | same (17575) |
| 130 | 2.63e-2 | same (5392) |
| 512 | 4.74e-2 | same (1281) |
| 1030 | 1.99e-2 | same (6252) |

All under the 5e-2 bar, which is Phase 9's logits-vs-golden bar, borrowed: the natural scale
of an engine-vs-engine comparison is the ~1e-2 amplification, and these sit at 2-5e-2 with the
argmax agreeing every time. The gate reproduced the 4.539e-2 exactly (deterministic) and
excluded Phase 11's two-launch mixes as the cause by switching it off: the gap then reads
4.97e-2, so the difference is inherent to the two paths and predates Phase 11. The remaining
candidate is a near-tie routing flip between the two own-router paths (the decode test's
own-router step sits at 1.0e-1 against the golden after its layer-10 flip); a forced-routing
form of this comparison is what settles it, and it needs a per-step routing golden at these
positions that does not exist yet. Reported as measured; the argmax agreement is the property
the odd-length rule uses.

**The speed assertion (gate 12 finding 2):** the generate test's bar was a third of the decode
test's number; the gate showed it non-vacuous (with the mixes shape off the loop went to 278
ms and the check failed) but blind to a 20% regression. It is now 210 ms against the loop's
own runs of 173-192 ms on this build (five runs, two sessions; the loop's spread is real and
larger than the decode test's 0.6 ms), so a 20% regression (~215 ms) fails.

**The GGUF path, measured by the gate, not argued:** a Qwen-1.5B GGUF loaded through the
untouched `Engine::load` on one card and generated correctly; `ds41_dir` requires a
`config.json` naming the arch, which no GGUF file path can satisfy.

**Concurrency (criterion 5's clause), demonstrated:** two chat requests fired one second apart
at the parallel-1 server ("Count from one to twelve in words…" and "List the seven days of the
week…", chat mode, temperature 0): both returned finish "stop" with the right text — "one, two,
three, … twelve" (23 tokens) and "Monday, Tuesday, … Sunday" (13 tokens) — and the server's log
shows the two generations one after the other (the second queued on the engine's gate, never
interleaved into the runtime's per-layer state); `/health` afterwards inflight 0, queued 0;
`/admin/shutdown` exited the process. (The `/health` sample taken during the overlap was lost
when the harness killed the driving script under the pinned run's memory pressure; the two
responses and the server log are the evidence.)

Refusals (criterion 7), exercised: a tool turn in the prompt format (error), an effort of 101
(error), tool definitions on the server route (`finish_reason: "error: …"`), images on the server
route (error). DSpark is not wired anywhere, so nothing to refuse.

## A note for the founder, not a Phase 12 finding (gate 12 pre-read, P4)

`ds41_encode_messages` concatenates the role markers with the raw message content, and the
route encodes the assembled prompt with `allow_special = true`, so a special-token literal
inside user content (the system marker, say) tokenises as that control token. This is how the
Python reference behaves too — criterion 2's byte-for-byte parity is correct as written — and
it is how every arch in this engine has always behaved (`engine.cpp`'s generate path and each
per-arch chat path encode the same way; `tokenizer.hpp` documents the trade and names the
alternative). Phase 12 changes the reach — one more externally callable route onto an encoder
that trusts its input — not the behaviour. The decision (escape role markers in user content at
the message boundary, or accept and document the exposure) is a product call across every
arch, raised here and left untouched in this phase.
