# HF-transformers per-layer reference-dump oracle

**Status:** harness built + format-validated on CPU (2026-06-12). Cross-check
against an engine GPU dump is the documented next step (deferred — GPU was held).

## What it is

`scripts/hf_oracle_dump.py` runs a model's **HF/PyTorch reference implementation**
on the **CPU** and writes per-layer residual-stream activations in the exact
binary format that `tools/diff_layers.sh` already consumes. `scripts/hf_oracle_dump.sh`
is a thin wrapper that runs the HF dump and then cosine-diffs it against an
existing engine dump.

## Why (the strategic unlock)

Today the engine validates a new architecture by cosine-diffing its per-layer
activations against **llama.cpp** dumps (`ie-llama-dump`). That makes us a
**follower**: llama.cpp must support the arch before we can validate ours.

The **real ground truth** is the HF/PyTorch reference implementation, which ships
**with every model on release day**. Validating against HF truth lets us bring up
**any new arch day-one** — the unlock for being **first-on-Intel-Arc** for hyped
flagship drops. Build it once, reuse it for every future arch.

## Format contract (reverse-engineered from the engine dumpers)

Sources: `src/model/dense_transformer.cpp` (≈L297–320, L546, L558),
`src/model/qwen35_dense.cpp` (≈L427–440), `tools/qwen35_dump.cpp`,
`tools/diff_layers.sh`.

| Item | Value |
|------|-------|
| Filenames | `${prefix}_L<NN>.bin` and `${prefix}_L<NN>.meta`, `NN` = `%02d` zero-padded |
| `.bin` payload | raw **little-endian float32**, **row-major**, shape **[T, H]**, **no header** (T = prompt tokens, H = hidden_size) |
| `.meta` payload | one ASCII line: `"T H\n"` |
| `L00` | residual stream **after the embedding** (pre layer 0) |
| `L01 .. L<n>` | residual stream **after** transformer layer `(k-1)` — i.e. `L<k>` is the output of decoder layer `k-1` |
| `L<n+1>` | output of the **final norm** (the vector fed to `lm_head`) |

`diff_layers.sh <mine_prefix> <ref_prefix> [MAX_SLOT]` loads each pair as fp32,
and reports `max_abs`, relative-Frobenius error, and **cosine similarity** per
slot. It tail-compares when row counts differ (llama.cpp slices the last layer to
the last token via `inp_out_ids`) and skips slots missing on **both** sides.

The engine residual stream is **fp16** (llama.cpp / HF keep fp32), so per-element
absolute error scales with activation magnitude — judge **relative-Frobenius /
cosine**, not absolute diffs. Gate: `cos_sim >= 0.9985` on every comparable slot
(same bar as `p3_parity_llama3.sh`). A wrong Q/K un-permute, `rope_freqs`, or norm
blows up at **L01**.

### HF → engine slot mapping

HF `model(..., output_hidden_states=True)` returns a tuple of `(n_layers + 1)`
tensors: `hidden_states[0]` = post-embedding, `hidden_states[k]` = output of
decoder layer `k-1`. Those map **1:1** onto `L00..L<n>`. The script then runs the
model's final norm (`model.model.norm` for Llama/Qwen2/Qwen3, `transformer.ln_f`
for GPT-2, etc. — located generically) on the last hidden state and writes it as
`L<n+1>`, matching the engine's final-norm slot.

## CPU / RAM constraints (HARD)

- **No GPU, no CUDA/XPU.** The engine holds the GPU under the project's
  GPU-exclusivity rule. The script pins **CPU** (`device_map="cpu"`, falling back
  to a plain load + `.to("cpu")` when `accelerate` is absent) and never imports an
  `ie-*` tool. It also warns-and-ignores if CUDA happens to be visible.
- **30 GB RAM ceiling on this box.** fp32 weights are 4 bytes/param, so a 1.5B
  model is ~6 GB resident plus activations. Use a **small model (≤1.5B)**, a
  **short prompt (≤8 tokens)**, and check `free -g` before loading. Do not OOM the
  box. For a large flagship you cannot fit in fp32 on CPU, run the HF dump on a
  machine with more RAM (or `--dtype bfloat16` to halve it) — the format is
  identical.

## Environment setup (CPU-only torch — small, CUDA-free)

```bash
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU wheel, no CUDA
pip install transformers
```

The CPU wheel avoids the multi-GB NVIDIA CUDA stack the default `pip install torch`
pulls. `accelerate` is **optional** (only needed for the literal `device_map="cpu"`
form; the script falls back without it). If torch/transformers are missing, the
script prints this exact hint and exits non-zero.

## Recommended workflow for validating a NEW arch

1. **Download** the HF repo for the model (the `config.json` + safetensors +
   tokenizer). This is the ground truth and is available on release day.
2. **Import to our format / GGUF** so the engine can load it:
   `ie import <hf_dir> ...` (AWQ/GPTQ/safetensors → GGUF; see the import tooling).
3. **Engine dump (GPU session, separate)** — one model on the GPU at a time:
   ```bash
   build/tools/ie-dense-dump  --gguf <model.gguf> --dump /tmp/eng/mine -p "The capital of France is"
   # or, for the qwen35 dense-hybrid:
   build/tools/ie-qwen35-dump --gguf <model.gguf> --dump /tmp/eng/mine -p "The capital of France is"
   ```
4. **HF reference dump (CPU session, this harness)** — same prompt:
   ```bash
   source ~/venv/bin/activate
   python scripts/hf_oracle_dump.py --model <hf_dir> \
       --prompt "The capital of France is" --out /tmp/hf/ref
   ```
   (Match tokenization: add `--no-special` if the engine tokenizer does **not**
   add BOS, so token IDs and T line up.)
5. **Cosine diff** the two dump dirs:
   ```bash
   tools/diff_layers.sh /tmp/eng/mine /tmp/hf/ref <MAX_SLOT>   # MAX_SLOT = n_layers+1
   ```
   or in one shot (steps 4+5), pointing at an **already-produced** engine dump:
   ```bash
   PY=~/venv/bin/python scripts/hf_oracle_dump.sh <hf_dir> /tmp/eng/mine <MAX_SLOT> "The capital of France is"
   ```
   `hf_oracle_dump.sh` runs only the **CPU** HF dump + the diff; it refuses to run
   if the engine dump is absent and **never** invokes an `ie-*` tool or the GPU.

Expect `cos_sim ~1.0` on every comparable slot. Divergence at `L01` = a
rope/permute/norm bug; divergence that grows with depth = an accumulating
numerical issue (often a wrong eps, scaling, or activation function).

## Worked example (the validation done while building this)

A tiny random-weights `Qwen3ForCausalLM` (4 layers, H=64) was dumped on CPU and
the files confirmed well-formed, then self-diffed:

```
$ python scripts/hf_oracle_dump.py --model <tiny_qwen3> \
      --prompt "The capital of France is" --out /tmp/hforacle/ref
  prompt='The capital of France is' tokens=[785, 6722, 315, 9625, 374] (T=5)
  hidden_size H=64, transformer layers=4
  wrote L00.bin (320 floats = 5 x 64) ... L05.bin (final-norm output)

$ tools/diff_layers.sh /tmp/hforacle/ref /tmp/hforacle/ref 5
  LAYER  max_abs   rel_fro   cos_sim     verdict
     00  0         0         1.000000    OK
     ...
     05  0         0         1.000000    OK
```

All slots `cos_sim=1.000000`, shapes `[5, 64]`, finite values — the format is
**drop-in compatible** with `diff_layers.sh`.

## Deferred (next GPU session)

The real cross-check — **HF dump vs engine GPU dump cos_sim ≈ 1.0** on a real
model — needs an engine dump produced on the GPU, which was held during the build
of this harness (GPU-exclusivity rule). Run steps 3–5 above in a GPU session on a
small real model (e.g. Qwen3-4B once imported to GGUF) to close the loop.

## CLI reference

```
scripts/hf_oracle_dump.py --model <hf_dir> --out <prefix>
    [--prompt "..."]      prompt to run (keep short; default "The capital of France is")
    [--layers N]          cap transformer layers emitted (0 = all)
    [--no-special]        do not add special/BOS tokens (match engine tokenizer)
    [--dtype float32|float16|bfloat16]   compute dtype (default float32)

scripts/hf_oracle_dump.sh <HF_MODEL_DIR> <ENGINE_DUMP_PREFIX> [MAX_SLOT] [PROMPT]
    env: PY=<python>  HF_OUT=<hf_prefix>  NOSPECIAL=1
```
