"""Layer-parity oracle: run ONE real qwen4exp decoder block on CPU, dump goldens.

    python run_block_oracle.py --layer L --tokens T --seed S --out out.npz [--with-ple]

Loads block L's weights from the UD-Q4_K_XL GGUF (inverting every converter
transform; see gguf_weights.py), instantiates Qwen4ExpTextDecoderLayer directly
(torch.float32, CPU, eval, no grad, deterministic), feeds a fixed-seed random WIDE
residual state [1, T, 4*2560] with position_ids arange(T) through the layer's full
forward path (hc_attn mix -> DeltaNet / QSA full attention -> hc combine -> hc_ffn
mix -> MoE(+shared) -> combine), and writes an .npz of inputs, output, and
intermediates.

Forward kwargs passed to the layer (mirrors Qwen4ExpTextModel.forward for a fresh
no-padding text batch):
  position_embeddings = Qwen4ExpTextRotaryEmbedding(cfg)(x, arange(T)[None]) ->
                        (cos, sin) each [1, T, 64] fp32   (text-only IMROPE)
  attention_mask      = eager 4D float causal mask [1,1,T,T] (0 visible / min above)
  conv_mask           = None  (no padding; apply_mask_to_padding_states is identity,
                        exactly what the full model computes for an all-ones mask)
  past_key_values     = None  (fresh prefill; DeltaNet uses the chunked rule)
  ple_input_ids       = the random token ids ([1,T]) when --with-ple, else None

--with-ple (only valid for --layer 1): a fixed-seed random token-id sequence over
vocab 248320 with ids[T//2] forced to EOS 248044 drives the PLE path; the 26.8 GiB
IQ4_NL n-gram table is gathered row-sparse from the GGUF (never materialized).
For layer 1 WITHOUT --with-ple, layer.ple is set to None so the dump covers the
pure DeltaNet+MoE path (documented in npz key `ple_enabled`).

npz keys are listed in README.md.  All floats fp32; [T,...] arrays have the batch
dim squeezed.
"""

from __future__ import annotations

import argparse
import contextlib
import sys

import numpy as np

import qwen4exp_ref as ref
import gguf_weights as gw

BIG_EMBED_THRESHOLD = 10_000_000


def _make_sparse_table(torch, nn, num_embeddings, embedding_dim):
    """Stands in for nn.Embedding(320001536, 160); dequantizes only gathered rows."""
    class _Table(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            # plain tensor attribute (NOT a Parameter): device anchor for the
            # `.weight.device` access in Qwen4ExpTextNGramEmbedding.forward
            self.weight = torch.zeros(1, embedding_dim)
            self._reader = None

        def forward(self, ids):
            assert self._reader is not None, "row reader not attached"
            flat = ids.reshape(-1).to(torch.long)
            uniq, inverse = torch.unique(flat, return_inverse=True)
            rows = torch.from_numpy(self._reader(uniq.numpy()))
            out = rows[inverse]
            return out.reshape(*ids.shape, self.embedding_dim)

    return _Table()


@contextlib.contextmanager
def sparse_embedding_patch(torch, holder):
    """Swap torch.nn.Embedding for a row-sparse stand-in when num_embeddings is huge.

    Only the PLE n-gram table crosses the threshold inside a decoder layer.
    """
    import torch.nn as nn

    real = nn.Embedding

    def factory(num_embeddings, embedding_dim, *a, **kw):
        if num_embeddings > BIG_EMBED_THRESHOLD:
            m = _make_sparse_table(torch, nn, num_embeddings, embedding_dim)
            holder["table"] = m
            return m
        return real(num_embeddings, embedding_dim, *a, **kw)

    nn.Embedding = factory
    try:
        yield
    finally:
        nn.Embedding = real


def build_layer(mod, cfg, store, layer):
    """Construct decoder layer `layer`, load GGUF weights, return (layer_module, holder)."""
    import torch

    holder = {}
    if layer == 1:
        with sparse_embedding_patch(torch, holder):
            lyr = mod.Qwen4ExpTextDecoderLayer(cfg, layer)
        holder["table"]._reader = store.ple_row_reader()
    else:
        lyr = mod.Qwen4ExpTextDecoderLayer(cfg, layer)

    sd, notes = gw.block_state_dict(store, layer)
    missing, unexpected = lyr.load_state_dict(sd, strict=False)
    allowed_missing = {
        "ple.ple_embedding.layer_multipliers",
        "ple.ple_embedding.ngram_heads_vocab_sizes",
        "ple.ple_embedding.ngram_heads_offsets",
    }
    bad_missing = [m for m in missing if m not in allowed_missing]
    assert not bad_missing, f"missing weights: {bad_missing}"
    assert not unexpected, f"unexpected weights: {unexpected}"

    if layer == 1:
        consts = gw.check_ple_constants(store, lyr.ple.ple_embedding)
        holder["ple_constants"] = consts

    lyr = lyr.float().eval()
    return lyr, notes, holder


def run(layer, tokens, seed, out_path, with_ple):
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    mod = ref.load_modeling()
    cfg = ref.build_text_config()
    store = gw.GgufStore()

    if with_ple and layer != 1:
        sys.exit("--with-ple is only valid for --layer 1 (ple_layer_ids=[2], 1-based)")

    lyr, inv_notes, holder = build_layer(mod, cfg, store, layer)
    kind = cfg.layer_types[layer]
    print(f"layer {layer} ({kind}) loaded; inversions:")
    for n in inv_notes:
        print("  ", n)

    T = tokens
    H = cfg.hidden_size
    HC = cfg.hc_count
    gen = torch.Generator().manual_seed(seed)
    # draw order is fixed: (1) wide state, (2) token ids -- do not reorder
    x_wide = torch.randn(1, T, HC * H, generator=gen, dtype=torch.float32)
    eos = cfg.eos_token_id if not isinstance(cfg.eos_token_id, list) else cfg.eos_token_id[0]
    ple_ids = torch.randint(0, cfg.vocab_size, (1, T), generator=gen)
    ple_ids[0, T // 2] = eos  # one EOS mid-sequence to exercise the n-gram reset

    if layer == 1 and not with_ple:
        lyr.ple = None  # dump the pure DeltaNet+MoE path
    ple_enabled = int(lyr.ple is not None) if layer == 1 else 0

    cos, sin = ref.make_position_embeddings(mod, cfg, T)
    mask = ref.make_full_attn_mask(T)
    position_ids = torch.arange(T)

    caps = {}

    def grab(name, index=None):
        def hook(_m, _inp, out):
            caps[name] = out if index is None else out[index]
        return hook

    hooks = [
        lyr.attn_hyper_connection.register_forward_hook(grab("hc_attn_mix", 0)),
        lyr.attn_hyper_connection.register_forward_hook(grab("hc_attn_inject_w", 2)),
        lyr.mlp_hyper_connection.register_forward_hook(grab("hc_ffn_mix", 0)),
        lyr.mlp_hyper_connection.register_forward_hook(grab("hc_ffn_inject_w", 2)),
        lyr.mlp.register_forward_hook(grab("moe_out")),
        lyr.mlp.gate.register_forward_hook(grab("router_logits", 0)),
        lyr.mlp.gate.register_forward_hook(grab("router_top_w", 1)),
        lyr.mlp.gate.register_forward_hook(grab("router_top_idx", 2)),
        lyr.mlp.shared_expert.register_forward_hook(grab("shared_expert_out")),
    ]
    if kind == "linear_attention":
        hooks.append(lyr.linear_attn.register_forward_hook(grab("mixer_out")))
    else:
        hooks.append(lyr.self_attn.register_forward_hook(grab("mixer_out", 0)))
    if ple_enabled:
        hooks.append(lyr.ple.register_forward_hook(grab("ple_out")))
        hooks.append(lyr.ple.ple_embedding.register_forward_hook(grab("ple_embed_E")))

    with torch.no_grad():
        y_wide = lyr(
            x_wide,
            position_embeddings=(cos, sin),
            attention_mask=mask,
            conv_mask=None,
            past_key_values=None,
            ple_input_ids=ple_ids if ple_enabled else None,
        )
    for h in hooks:
        h.remove()

    def sq(t):
        a = t.detach().to(torch.float32).cpu().numpy()
        return a[0] if a.ndim >= 2 and a.shape[0] == 1 else a

    out = {
        "layer": np.int64(layer),
        "tokens": np.int64(T),
        "seed": np.int64(seed),
        "layer_type": np.bytes_(kind.encode()),
        "ple_enabled": np.int64(ple_enabled),
        "x_wide": sq(x_wide),
        "position_ids": position_ids.numpy().astype(np.int64),
        "cos": sq(cos),
        "sin": sq(sin),
        "y_wide": sq(y_wide),
        "hc_attn_mix": sq(caps["hc_attn_mix"]),
        "hc_attn_inject_w": sq(caps["hc_attn_inject_w"]),
        "mixer_out": sq(caps["mixer_out"]),
        "hc_ffn_mix": sq(caps["hc_ffn_mix"]),
        "hc_ffn_inject_w": sq(caps["hc_ffn_inject_w"]),
        "moe_out": sq(caps["moe_out"]),
        "router_logits": caps["router_logits"].detach().float().numpy(),
        "router_top_w": caps["router_top_w"].detach().float().numpy(),
        "router_top_idx": caps["router_top_idx"].detach().numpy().astype(np.int64),
        "shared_expert_out": caps["shared_expert_out"].detach().float().numpy(),
    }
    if ple_enabled:
        out["ple_input_ids"] = ple_ids[0].numpy().astype(np.int64)
        out["ple_out"] = sq(caps["ple_out"])
        out["ple_embed_E"] = sq(caps["ple_embed_E"])
        c = holder["ple_constants"]
        out["ple_layer_multipliers"] = np.array(c["layer_multipliers"], dtype=np.uint64)
        out["ple_head_offsets"] = np.array(c["head_offsets"], dtype=np.int64)
        out["ple_head_vocab_sizes"] = np.array(c["head_vocab_sizes"], dtype=np.int64)
        out["ple_eos_token_id"] = np.int64(c["eos"])

    np.savez(out_path, **out)

    bad = []
    for k, v in out.items():
        a = np.asarray(v)
        if a.dtype.kind == "f":
            if not np.isfinite(a).all():
                bad.append(f"{k}: non-finite")
            elif a.size > 4 and float(a.std()) == 0.0:
                bad.append(f"{k}: zero variance")
    status = "OK" if not bad else "DEGENERATE: " + "; ".join(bad)
    print(f"wrote {out_path}  [{status}]")
    for k in ("x_wide", "hc_attn_mix", "mixer_out", "hc_ffn_mix", "moe_out", "y_wide"):
        a = out[k]
        print(f"  {k:16s} {str(list(a.shape)):16s} sum={float(np.float64(a).sum()):+.6e} "
              f"std={float(a.std()):.6e}")
    if ple_enabled:
        for k in ("ple_embed_E", "ple_out"):
            a = out[k]
            print(f"  {k:16s} {str(list(a.shape)):16s} sum={float(np.float64(a).sum()):+.6e} "
                  f"std={float(a.std()):.6e}")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-ple", action="store_true")
    args = ap.parse_args()
    sys.exit(run(args.layer, args.tokens, args.seed, args.out, args.with_ple))


if __name__ == "__main__":
    main()
