# V4.1 port — Phase 12 gate criteria: usable end to end (written BEFORE the build)

**Starting line (Phase 11, 0111bf1):** prefill 281 tok/s at 2048 tokens, decode 5.13 tok/s at a
2048-token context, both on the resident two-card runtime, both correct against the reference's
goldens — and reachable only through test binaries that take token ids from Python dumps. The
founder's standing instruction is "get v4.1 up and going and optimize for speed and
performance"; "up and going" is not met while a person cannot type a prompt at it.

## Why this phase, and not the next decode lever

docs/30 names the next speed levers: a batched verify (DSpark / prompt lookup) and the
FP8-resident dense set. Sized before choosing (docs/30, the reference `model.py:129`): the
reference ships only the DSpark block's forward pass, no speculative loop, so the drafter's
generation procedure would be built from the report; and an MoE verify of k rows routes to
nearly k times the experts (top-6 of 384 overlaps little across rows), so the fetch scales with
the draft length and the gain is bounded by the acceptance rate — roughly 1.5x at 60% acceptance
by the arithmetic in docs/30, not the 2-3x dense models get. Real, but a multi-day phase with
its own golden, on a model nobody can yet use. The usable path first; the verify phase after,
with a working generate loop to build on. (A founder redirect reverses the order without loss:
nothing in this phase is a prerequisite of the other.)

## Scope, in order; each item verified before the next

1. **Tokenizer.** A loader for the checkpoint's `tokenizer.json` (HF tokenizers BPE, vocab
   128,000, 127,741 merges, byte-level decoder, the pre-tokenizer Sequence: `\p{N}{1,3}` runs,
   CJK runs, then the DeepSeek-V3/V4 split) feeding the engine's existing `Tokenizer` — the
   BPE and the "joyai-llm" three-pass DeepSeek split already exist for V4-Flash (GGUF); this
   phase adds the JSON source and any special tokens V4.1 adds (`<｜System｜>`, the DSML tags
   with the leading space, the noise token). Gate: `ie-ds41-tokenizer-test` encodes the held-out
   corpus text (`docs/*.md`, the 65,536-token corpus of Phase 8b) and the pp2048 text and
   compares ids with the Python reference (HF `tokenizers` through the checkpoint's
   `tokenizer.json`), token for token; decode(encode(x)) == x on the same texts; the special
   tokens round-trip.
2. **Prompt format.** `encode_messages` of the checkpoint's `encoding/encoding.py` ported for
   text: system, user, assistant turns, the thinking/non-thinking modes, the reasoning-effort
   prefix (numeric budget, index 0 only), mid-conversation `<｜System｜>`, and the generation
   header. Tool calls (DSML blocks) and images are out of scope here and refused, not faked.
   Gate: the rendered prompt for each of the reference's `encoding/tests` text cases equals the
   Python reference's byte for byte (the test cases ship with the checkpoint).
3. **Generate loop.** On `Ds41Forward` (resident, two cards): prefill the prompt (replay ON as
   deployed), greedy or sampled (temperature, top-p, top-k, repetition penalty — the engine's
   existing `sampling.cpp`), a step per token, streaming detokenisation with the byte-level
   decoder's partial-UTF-8 handling, the stop conditions (eos, `<｜end▁of▁sentence｜>`, a
   token budget, stop strings). Gate: greedy generation from the golden prompt reproduces the
   golden's continuation tokens (docs/25's four) and its text; a 2048-token prompt followed by
   64 greedy tokens equals the replay test's teacher-forced tokens where those exist; sampling
   at temperature 0 equals greedy; the loop's ms/token equals the decode test's within run
   noise (no regression hidden in the loop).
4. **CLI.** `ie-ds41-run <model> [--prompt|--chat] [--n] [--temp ...]`: loads the resident
   runtime (the held-out ranking from a file argument or a default location beside the model),
   prints the placement summary and the load time, then streams text. Gate: a founder-style
   session — a question, a 2,000-token pasted document with a question about it, a
   multi-turn exchange — produces coherent text with the reported tok/s, and `Ctrl-C` unloads
   cleanly (the three-point unload check's residue bar).
5. **OpenAI route.** `/v1/chat/completions` (streaming and not) on the engine's server layer
   (`openai_proto`, `openai_server`) for V4.1 — either the `Ds4Bundle` pattern inside `Engine`
   or a dedicated server binary like `glm5_server`, whichever needs fewer changes to the
   GGUF-centric `Engine` (decided in the build, recorded in the results). Gate: the server's
   existing conformance tests pass for the new arch; a `curl` chat request returns the same
   text as the CLI for the same prompt at temperature 0; concurrent requests are refused or
   queued, never interleaved into one state.

## Pass criteria

1. Tokenizer parity token for token on the two corpora (0 differences), round-trip exact.
2. Prompt-format parity byte for byte on the reference's text test cases.
3. The generate loop's greedy tokens equal the decode test's and the replay test's; sampling
   at temperature 0 equals greedy; ms/token within run noise of the decode test.
4. The CLI session, transcribed in the results doc, with load time, placement and tok/s.
5. The server route: conformance tests PASS; CLI == server at temperature 0.
6. **No regression:** the decode, forward, resident, replay and tier tests PASS on the same
   build; unload unchanged.
7. **Nothing faked:** tool calls, images and DSpark are refused with a message, not stubbed.

## Explicitly NOT in this phase

Speculative decoding (Phase 13 candidate, with the batched verify path as its first step),
the FP8-resident dense set, tool-call and image prompt formats, the vision tower, batching.
