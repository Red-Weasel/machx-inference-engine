#!/usr/bin/env python3
"""hf_oracle_dump.py — HF/PyTorch per-layer reference-dump oracle (CPU only).

Strategic purpose
-----------------
The engine validates a NEW architecture by cosine-diffing its per-layer
residual-stream activations against a reference dump (tools/diff_layers.sh).
Historically the reference was llama.cpp (tools/ie-llama-dump), which makes us a
*follower*: llama.cpp must support the arch first. The REAL ground truth is the
HF/PyTorch reference implementation that ships WITH every model on release day.
This script produces a reference dump from HF transformers in the EXACT binary
format tools/diff_layers.sh consumes, so we can validate ANY new arch against HF
truth on day one — the unlock for being first-on-Intel-Arc for flagship drops.

Format contract (reverse-engineered from the engine dumpers — see
docs/hf_reference_oracle.md):

  Files:  ${prefix}_L<NN>.bin   +   ${prefix}_L<NN>.meta   (NN = zero-padded %02d)
  .bin :  raw little-endian float32, row-major, shape [T, H]
          (T = prompt tokens, H = hidden_size). No header.
  .meta:  one ASCII line "T H\n".

  Slot convention (matches src/model/dense_transformer.cpp & qwen35_dense.cpp):
    L00            = residual stream AFTER the embedding (pre layer 0)
    L01 .. L<n>    = residual stream AFTER transformer layer (i-1), i.e. L<k> is
                    the output of decoder layer k-1
    L<n+1>         = output of the FINAL norm (the thing fed to lm_head)

  This maps 1:1 onto HF `output_hidden_states=True`, which returns a tuple of
  (n_layers + 1) tensors: hidden_states[0] = post-embedding, hidden_states[k] =
  output of layer k-1. We emit those as L00..L<n>, then run the model's final
  norm on the last hidden state and emit it as L<n+1>.

CPU / RAM constraints
---------------------
Loads with device_map="cpu", torch_dtype=torch.float32 (NO CUDA/XPU — the engine
holds the GPU under the project's GPU-exclusivity rule). Use a SMALL model and a
SHORT prompt; check `free -g` first. fp32 weights are 4 bytes/param, so a 1.5B
model is ~6 GB resident plus activations.
"""
import argparse
import os
import struct
import sys


def _die_imports(exc):
    sys.stderr.write(
        "ERROR: could not import torch / transformers: %s\n\n"
        "This harness needs PyTorch (CPU build) and HF transformers.\n"
        "Install into a venv (CPU-only torch keeps it small and CUDA-free):\n\n"
        "    python3 -m venv ~/venv && source ~/venv/bin/activate\n"
        "    pip install --index-url https://download.pytorch.org/whl/cpu torch\n"
        "    pip install transformers\n\n"
        "Then re-run with that venv active.\n" % exc
    )
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser(
        description="HF/PyTorch per-layer reference dump (CPU) in the engine's "
                    "diff_layers.sh format.")
    ap.add_argument("--model", required=True,
                    help="HF model dir (config.json + weights + tokenizer).")
    ap.add_argument("--prompt", default="The capital of France is",
                    help="Prompt to run the forward on (keep it SHORT, <=8 tok).")
    ap.add_argument("--out", required=True,
                    help="Output dump prefix, e.g. /tmp/hf/ref → /tmp/hf/ref_L00.bin")
    ap.add_argument("--layers", type=int, default=0,
                    help="Cap number of transformer layers emitted (0 = all). "
                         "L00..L<layers> + final-norm slot are written.")
    ap.add_argument("--no-special", action="store_true",
                    help="Do NOT add special tokens when encoding the prompt "
                         "(match the engine tokenizer if it does not add BOS).")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"],
                    help="Compute dtype. Default float32 (CPU ground truth). "
                         "The engine residual stream is fp16, but the reference "
                         "should be fp32 — diff_layers.sh judges relative error.")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # ModuleNotFoundError or a broken install
        _die_imports(exc)

    # --- HARD: CPU only. Never touch CUDA/XPU (GPU exclusivity rule). ---
    if torch.cuda.is_available():
        sys.stderr.write(
            "NOTE: CUDA is visible but this harness pins CPU on purpose "
            "(GPU exclusivity). Forcing device_map='cpu'.\n")
    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    sys.stderr.write("loading tokenizer: %s\n" % args.model)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    sys.stderr.write("loading model (CPU, %s)...\n" % args.dtype)
    # device_map="cpu" is the documented form but it requires `accelerate`.
    # Fall back to a plain load + .to("cpu") so the harness works without it.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, device_map="cpu", trust_remote_code=True)
    except (ValueError, ImportError) as exc:
        if "accelerate" not in str(exc).lower():
            raise
        sys.stderr.write("  (accelerate not installed; loading without "
                         "device_map and pinning CPU manually)\n")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, trust_remote_code=True)
        model = model.to("cpu")
    model.eval()

    enc = tok(args.prompt, return_tensors="pt",
              add_special_tokens=not args.no_special)
    input_ids = enc["input_ids"]  # [1, T]
    T = int(input_ids.shape[1])
    ids_list = input_ids[0].tolist()
    sys.stderr.write("prompt=%r tokens=%s (T=%d)\n" % (args.prompt, ids_list, T))

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)

    # hidden_states: tuple length (n_layers + 1).
    #   [0]  = post-embedding residual                  -> L00
    #   [k]  = output of decoder layer k-1              -> L<k>
    hs = out.hidden_states
    n_layers = len(hs) - 1
    H = int(hs[0].shape[-1])
    sys.stderr.write("hidden_size H=%d, transformer layers=%d\n" % (H, n_layers))

    cap = n_layers if args.layers <= 0 else min(args.layers, n_layers)

    def write_slot(idx, tensor):
        # tensor: [1, T, H] (or [T, H]) -> fp32 row-major [T, H]
        t = tensor
        if t.dim() == 3:
            t = t[0]
        arr = t.detach().to(torch.float32).contiguous().view(-1).tolist()
        n = len(arr)
        finite = all(v == v and abs(v) != float("inf") for v in arr[:min(n, 4096)])
        bin_path = "%s_L%02d.bin" % (args.out, idx)
        with open(bin_path, "wb") as f:
            f.write(struct.pack("<%df" % n, *arr))
        with open("%s_L%02d.meta" % (args.out, idx), "w") as f:
            f.write("%d %d\n" % (int(t.shape[0]), int(t.shape[1])))
        return n, finite

    # L00 .. L<cap>
    for idx in range(0, cap + 1):
        n, finite = write_slot(idx, hs[idx])
        if not finite:
            sys.stderr.write("  WARNING: L%02d has non-finite values!\n" % idx)
        sys.stderr.write("  wrote L%02d.bin  (%d floats = %d x %d)%s\n" % (
            idx, n, T, H, "" if finite else "  [NON-FINITE]"))

    # Final-norm slot L<n+1>: run the model's final RMS/Layer norm on the LAST
    # hidden state (output of the last decoder layer). We locate the norm
    # generically so this works across Llama/Qwen/etc.
    final_norm = _find_final_norm(model)
    if final_norm is not None and cap == n_layers:
        with torch.no_grad():
            normed = final_norm(hs[n_layers].to(dtype))
        idx = n_layers + 1
        n, finite = write_slot(idx, normed)
        sys.stderr.write("  wrote L%02d.bin  (final-norm output, %d floats)%s\n" % (
            idx, n, "" if finite else "  [NON-FINITE]"))
    else:
        if cap != n_layers:
            sys.stderr.write("  (skipped final-norm slot: --layers capped run)\n")
        else:
            sys.stderr.write("  WARNING: could not locate final norm module; "
                             "skipped L%02d (final-norm slot).\n" % (n_layers + 1))

    # Greedy argmax of the last token, for the greedy-parity check
    # (mirrors ie-qwen35-dump / ie-dense-dump argmax line).
    logits = out.logits[0, -1]  # [vocab]
    top = torch.topk(logits.to(torch.float32), 5)
    sys.stderr.write("argmax: id=%d logit=%.3f piece=%r\n" % (
        int(top.indices[0]), float(top.values[0]),
        tok.decode([int(top.indices[0])])))
    for r in range(5):
        sys.stderr.write("  top%d id=%-7d logit=%8.3f %r\n" % (
            r + 1, int(top.indices[r]), float(top.values[r]),
            tok.decode([int(top.indices[r])])))

    max_slot = (n_layers + 1) if (final_norm is not None and cap == n_layers) else cap
    sys.stderr.write("dumped L00..L%02d to %s_LNN.bin  (diff_layers.sh MAX_SLOT=%d)\n"
                     % (max_slot, args.out, max_slot))


def _find_final_norm(model):
    """Best-effort locate the post-decoder final norm module (pre lm_head).

    Covers the common HF layouts: model.model.norm (Llama/Qwen2/Qwen3),
    model.transformer.ln_f (GPT-2), model.model.final_layernorm (some), etc.
    Returns a callable nn.Module or None.
    """
    candidates = [
        ("model", "norm"),
        ("model", "final_layernorm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
    ]
    for outer, inner in candidates:
        o = getattr(model, outer, None)
        if o is not None:
            nrm = getattr(o, inner, None)
            if nrm is not None:
                return nrm
    # Fallback: a top-level .norm / .ln_f
    for name in ("norm", "ln_f", "final_layernorm"):
        nrm = getattr(model, name, None)
        if nrm is not None:
            return nrm
    return None


if __name__ == "__main__":
    main()
