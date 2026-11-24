# machx-train

The command Dream's Train view calls to fine-tune a local model on this machine's Intel Arc cards. Installed as
`~/.local/bin/machx-train` (a launcher for `machx_train.py`). Python environment: `~/models/inkling2-train/venv`
(torch 2.15 + XPU, transformers 5.24 with the publisher's Inkling 2 code, unpatched).

## Contract (Dream `docs/training.md`)

```
machx-train --probe --model PATH
machx-train --model PATH --data FILE.jsonl --out NEW_FOLDER --metrics FILE.jsonl
            [--eval FILE.jsonl] [--max-steps N] [--max-examples N]
            [--cards 0,1] [--max-len 12288] [--resume] [--tools-sample SAMPLE.json]
```

- **Probe**: no model load. The last stdout line is `{"arch", "supported", "reason", "needs_gib_per_card"}`. PATH is a
  checkpoint folder (`config.json`) or a GGUF file (its header).
- **Run**: exit 0 and NEW_FOLDER is a complete model folder; otherwise non-zero and the last stderr line says why in
  plain words. Nothing from the data is ever printed or logged.
- **SIGTERM** (to the process group): the example under way finishes, that partial step is dropped, and the folder is
  written from the steps completed. Stopped before the first completed step: no folder, non-zero.
- **Metrics**: one JSON object per line: `step`, `loss`, `learning_rate`, `elapsed_seconds`, `tokens_per_second`,
  `eta_seconds`, and on evaluation steps `eval_loss`, `eval_success_rate`.
- `eval_success_rate` is **token accuracy**: the share of the trained tokens of the held-out assistant turns whose
  greedy next token is right. It is not a task success rate.

## Supported

Inkling 2 Mini (`inkling2_loop`) from its checkpoint folder, including a folder with run-time ablation files. Anything
else is answered `supported: false` ("no trainer for <arch> yet").

What it trains: LoRA rank 16, alpha 32 on the attention q / k / v / o of all 24 layers, folded into the stored BF16
weights in the forward, so the trained model is the exported model. AdamW 2e-4, 10 warm-up steps, cosine to 10 %,
8 examples a step, one pass over the data unless capped. A held-out set (`--eval`, else 3 % of the sessions, 8 to 32,
whole sessions; none below 40 sessions) is scored about 10 times; training stops after 3 evaluations without a 0.5 %
improvement and the folder gets the best evaluation's weights. Checkpoints every 15 minutes and at each evaluation in
`NEW_FOLDER.work` (removed on success; `--resume` continues from it).

## Data

One JSON object per line with `messages` (Dream's recorded shape: inline `<think>`, `<tool_call>` XML, role `tool`,
`weight` 0/1 on assistant turns; structured `reasoning_content` / `tool_calls` are also read).

- **Rendering** is the model's own chat template. Tool declarations: the record's `tools`, else the `tools` array of a
  Dream request sample (`python -m dream.request_sample`; default: the newest in
  `/media/<user>/Secondary Drive/inkling2-train-exports/dream-sample`). A Qwen tool preamble left in a recorded system
  page is removed. Thinking effort is the template default (Dream's request sets none).
- **Tool-result names**: each result is named after the call it answers (unique ids). Sessions recorded before engine
  commit 9c72d54 carry repeated ids (`call_chatcmpl_N`); the training render does not reproduce their wrong names.
- **Loss** only on the tokens the model writes for assistant turns with weight 1.
- **Windowing** (default 12,288 tokens): the system page and first user message stay; a longer session drops its
  oldest whole exchanges and becomes the fewest windows that cover its trained turns. A trained turn that cannot fit
  even with no history is cut at the limit (counted in the run log). Records that are prefixes of a longer record of
  the same session are merged into it. The page prefix is not cached: its keys and values depend on the adapters
  being trained.

## Measured (2 x Arc Pro B70, Inkling-2-Mini-abliterated, 2026-11-23)

| Tokens per example | One card | Peak card memory |
|---|---|---|
| 1,024 | 6.6 s, 155 tok/s | 20.7 GiB |
| 4,096 | 21.6 s, 190 tok/s | 22.7 GiB |
| 8,192 | 42.8 s, 191 tok/s | 25.7 GiB |
| 12,288 | 67.9 s, 181 tok/s | 29.2 GiB |

Two cards, real run on about 10,000-token sessions: 365 to 373 tok/s. One card in a real run of the same data: 151 tok/s
(below the single-example bench above; the cause was not investigated). Host memory: 66 GiB peak per load (staggered),
24 GiB after it. Time for one pass is the data's rendered tokens divided by the speed: about 1.33 M tokens an hour.

## Correctness notes

- The router's picks are stored per layer call and replayed when gradient checkpointing recomputes a layer.
- The reference's expert combine (`index_add`) is atomic on XPU: identical runs gave gradient cosine 0.84 to 0.91. The
  deterministic combine here makes repeated runs in one process identical in the forward.
- Between two PROCESSES the card's fp32 router matmul still differs in the last bits at long lengths, which can flip
  near-tied expert picks, so card logits of one model are not reproducible bit for bit across processes. The export is
  therefore verified on the CPU, where the folder and the adapter-applied forward are bit-identical (`inkling2/verify.py`).
- An abliterated folder's `ie_ablation.json` is applied in the training forward as the engine applies it.

## Adding an architecture

1. Add it to `SUPPORTED` in `machx_train.py` with its measured card and host memory.
2. Give it a package like `inkling2/`: model load + adapters (`model.py`), rendering (`data.py`), the per-card process
   (`worker.py`), export (`export.py`), and dispatch to its worker in `run()`.
3. Before trusting a gradient on the card: repeat one forward / backward (`det.py`), compare with the CPU
   (`bench.py`), check for routing or other discrete choices that a checkpoint recompute can change.
