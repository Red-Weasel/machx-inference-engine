#!/usr/bin/env python3
"""Hy4-flash mini-model oracle (docs/hy4flash/00_PORT_AND_ABLITERATION_PLAN.md).

Builds a tiny random Hy4-flash TEXT model from the reference implementation (transformers PR #50903), runs it on CPU in
fp32 and writes what the engine's mini gate compares against:

  python mini_model.py build <out_dir> [--tokens 96] [--seed 0] [--window 24] [--group-topk 6] [--no-engram]
      <out_dir>/config.json          the release's layout ({"model_type": "hy_v4_flash", "text_config": {...}})
      <out_dir>/model.safetensors    the weights under the RELEASED checkpoint's names, BF16 (what the engine binds)
      <out_dir>/oracle/ids.i32       the token ids fed
      <out_dir>/oracle/logits.f32    [T, vocab] float32
      <out_dir>/oracle/streams.f32   [layers, T, hc, hidden] float32: the residual streams entering each layer
      <out_dir>/oracle/routes.i32    [MoE layers, T, top_k] the experts each row picked (ascending)
      <out_dir>/oracle/route_margin.f32  [MoE layers, T] score of the last expert picked minus the first one not picked
      <out_dir>/oracle/recall.i32    [indexed layers that ran the indexer, T, group_topk] the groups each row recalled
      <out_dir>/oracle/meta.json     shapes + the settings above
  python mini_model.py oracle <out_dir> [--tokens 96] [--seed 0]
      the oracle files again from an existing <out_dir>/model.safetensors (no new weights)
  python mini_model.py keys <real model.safetensors.index.json>
      compares the tensor-name families of the reference module with the released checkpoint's index

Real widths (hidden 3072, latent 1536, the MLA ranks), tiny depth, expert count and tables. The defaults make every
mechanism bite inside 96 tokens: the sliding window (24), the group selection of the indexed layers (6 groups of 4
tokens), n-gram tokens past the lookup table. Every weight is a value BF16 holds exactly and every routed expert is on
the MXFP4 grid, so the engine (BF16 -> fp16 dense, 4-bit expert pack) computes with the SAME numbers as this fp32 run.

Run with the reference venv:  ~/models/hy4flash-reference/venv/bin/python mini_model.py ...
"""
import argparse, json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy4flash_common import mxfp4_quantize, release_name  # noqa: E402
from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashTextConfig  # noqa: E402
from transformers.models.hy_v4_flash.modeling_hy_v4_flash import HYV4FlashTextModel  # noqa: E402

PRIMES = [10007, 10009, 10037, 10039, 10061, 10067, 10069, 10079, 10091, 10093, 10099, 10103, 10111, 10133, 10139, 10141,
          10151, 10159, 10163, 10169, 10177, 10181, 10193, 10211, 10223, 10243, 10247, 10253, 10259, 10267, 10271, 10273]

def mini_config(window=24, group_topk=6, engram=True):
    # 6 layers = the release's opening pattern: indexed, indexed, sliding x3, indexed; layer 0 has the dense FFN.
    cfg = dict(vocab_size=4096, num_hidden_layers=6, n_routed_experts=64, num_experts_per_tok=16,
               pad_token_id=2, bos_token_id=0, eos_token_id=1, engram_pad_id=2, engram_lookup_table_size=4000,
               engram_layer_ids=[2, 5],                                   # applied at layers 1 and 4
               engram_layer_head_vocab_sizes={"2": [PRIMES[0:8], PRIMES[8:16]], "5": [PRIMES[16:24], PRIMES[24:32]]},
               max_position_embeddings=8192, sliding_window=window, compressed_indexer_group_topk=group_topk,
               swiglu_limit=2.0)                                          # low enough that the experts' clamp bites
    if not engram: cfg.update(engram_layer_ids=None, engram_layer_head_vocab_sizes=None)
    return HYV4FlashTextConfig(**cfg)

def randomize(model, seed):
    """Seeded, non-degenerate weights for everything the checkpoint stores (never the derived buffers: rope frequencies,
    table offsets): every 1-D tensor off its identity value, experts different from each other, a lookup table that is
    not the identity. Then: every float rounded to BF16, every routed expert put on the MXFP4 grid."""
    g = torch.Generator().manual_seed(seed)
    stored = set(model.state_dict().keys())
    with torch.no_grad():
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            if name not in stored: continue
            if name.endswith("lookup_table"):
                p.copy_(torch.randint(0, 3000, p.shape, generator=g, dtype=torch.int32)); continue
            if not torch.is_floating_point(p): continue
            if p.ndim == 1:
                base = 1.0 if ("norm" in name or "gamma" in name or "scale" in name) and "base" not in name and "bias" not in name else 0.0
                p.copy_(base + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) * (1.0 / p.shape[-1]) ** 0.5)
        for name, p in model.named_parameters():
            if name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj"):
                for e in range(p.shape[0]): p[e].copy_(mxfp4_quantize(p[e])[2])
            p.copy_(p.to(torch.bfloat16).float())
        for name, b in model.named_buffers():
            if name in stored and torch.is_floating_point(b): b.copy_(b.to(torch.bfloat16).float())

def families(names):
    return sorted({re.sub(r"\.\d+", ".N", n) for n in names})

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("out")
    b.add_argument("--tokens", type=int, default=96); b.add_argument("--seed", type=int, default=0)
    b.add_argument("--window", type=int, default=24); b.add_argument("--group-topk", type=int, default=6)
    b.add_argument("--no-engram", action="store_true"); b.add_argument("--image", default="")
    orc = sub.add_parser("oracle"); orc.add_argument("out")
    orc.add_argument("--tokens", type=int, default=96); orc.add_argument("--seed", type=int, default=0)
    orc.add_argument("--image", default="", help="start,rows,cols: an image span of rows * cols rows (cols counts the newline column) at token `start`")
    k = sub.add_parser("keys"); k.add_argument("index")
    a = ap.parse_args()
    torch.manual_seed(0)
    if a.cmd == "keys":
        model = HYV4FlashTextModel(mini_config())
        mine = {re.sub(r"\.\d+", ".N", release_name(n)) for n in model.state_dict().keys()}
        idx = json.load(open(a.index)); real_all = list(idx.get("weight_map", idx).keys())
        real = set(families(real_all))
        print("real checkpoint:", len(real_all), "tensors,", len(real), "families")
        print("families only in the mini text model (after the renames):", sorted(mine - real))
        only_real = sorted(real - mine)
        print(len(only_real), "families only in the real checkpoint (vision, drafter, head expected):")
        for n in only_real: print("   ", n)
        return
    if a.cmd == "build":
        cfg = mini_config(a.window, a.group_topk, not a.no_engram); model = HYV4FlashTextModel(cfg).eval().float()
        randomize(model, a.seed)
        head_w = (torch.randn((cfg.vocab_size, cfg.hidden_size), generator=torch.Generator().manual_seed(a.seed + 1)) * cfg.hidden_size ** -0.5).to(torch.bfloat16).float()
        os.makedirs(a.out, exist_ok=True)
        from safetensors.torch import save_file
        sd = {}
        for name, v in list(model.state_dict().items()) + [("lm_head.weight", head_w)]:
            sd[release_name(name)] = (v.to(torch.bfloat16) if torch.is_floating_point(v) else v).contiguous()
        save_file(sd, os.path.join(a.out, "model.safetensors"))
        json.dump({"model_type": "hy_v4_flash", "architectures": ["HYV4FlashForConditionalGeneration"], "dtype": "bfloat16",
                   "text_config": cfg.to_dict()}, open(os.path.join(a.out, "config.json"), "w"), indent=1)
        print(f"wrote {len(sd)} tensors, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    else:
        from safetensors.torch import load_file
        tc = json.load(open(os.path.join(a.out, "config.json")))["text_config"]
        cfg = HYV4FlashTextConfig(**{k: v for k, v in tc.items() if k not in ("model_type", "transformers_version", "architectures")})
        model = HYV4FlashTextModel(cfg).eval().float()
        sd = load_file(os.path.join(a.out, "model.safetensors"))
        back = {release_name(k): k for k in model.state_dict().keys()}
        head_w = sd.pop("lm_head.weight").float()
        missing = model.load_state_dict({back[k]: (v.float() if torch.is_floating_point(v) else v) for k, v in sd.items()}, strict=True)
        print("loaded", len(sd), "tensors;", missing)
    run_oracle(model, head_w, cfg, a.out, a.tokens, a.seed, a.image)

def run_oracle(model, head_w, cfg, out_dir, tokens, seed, image=""):
    # nn.RMSNorm built without an eps (the n-gram layer's norms) takes the epsilon of the dtype it runs in. The release
    # runs in bf16 (2^-7); this fp32 run is pinned to the same value so the oracle is the released model's arithmetic.
    n_eps = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.RMSNorm) and mod.eps is None: mod.eps = 2.0 ** -7; n_eps += 1
    ids = torch.randint(3, cfg.vocab_size, (1, tokens), generator=torch.Generator().manual_seed(seed + 2))
    streams, routes, margins, recalls, recall_layers = [], [], [], [], []
    hooks = [layer.register_forward_pre_hook(lambda m, args: streams.append(args[0].detach().float().clone())) for layer in model.layers]
    def route_hook(m, args, out):
        logits, _, idx = out
        choice = logits.sigmoid() + m.e_score_correction_bias
        top = choice.topk(m.top_k + 1, dim=-1).values
        routes.append(idx.sort(dim=-1).values.int().clone()); margins.append((top[:, m.top_k - 1] - top[:, m.top_k]).float().clone())
    for li, layer in enumerate(model.layers):
        if hasattr(layer.mlp, "gate"): hooks.append(layer.mlp.gate.register_forward_hook(route_hook))
        ix = layer.self_attn.compressed_indexer
        if ix is not None:
            def recall_hook(m, args, out, li=li):       # (returns None: a hook's return value would replace the output)
                recalls.append(out[0].int().clone()); recall_layers.append(li)
            hooks.append(ix.register_forward_hook(recall_hook))
    with torch.no_grad():
        if image:
            # An image span, as HYV4FlashModel feeds one: the rows carry the image token id, their embeddings are the
            # projector's vectors (random here), their positions are 3-D (t = start, h = start + row, w = start + column;
            # the text continues one past the largest), and they own no n-gram.
            st, R, C = (int(v) for v in image.split(","))
            img_id = cfg.vocab_size - 2
            ids[0, st:st + R * C] = img_id
            pos = torch.zeros((3, 1, tokens), dtype=torch.long); cur = 0; t = 0
            while t < tokens:
                if t == st:
                    for r in range(R):
                        for c in range(C): pos[:, 0, t] = torch.tensor([cur, cur + r, cur + c]); t += 1
                    cur = cur + max(R - 1, C - 1) + 1
                else: pos[:, 0, t] = cur; cur += 1; t += 1
            emb = model.embed_tokens(ids).clone()
            feats = torch.randn((R * C, cfg.hidden_size), generator=torch.Generator().manual_seed(seed + 7)) * float(model.embed_tokens.weight.std())
            emb[0, st:st + R * C] = feats
            is_embed = torch.zeros(tokens, dtype=torch.uint8); is_embed[st:st + R * C] = 1
            ngram = model.build_ngram_tokens(ids, pos) if cfg.engram_layer_ids else None
            mask = (ids != img_id) if cfg.engram_layer_ids else None
            out = model(inputs_embeds=emb, position_ids=pos, ngram_tokens=ngram, engram_mask=mask, use_cache=False)
        else:
            ngram = model.build_ngram_tokens(ids, None) if cfg.engram_layer_ids else None
            mask = torch.ones_like(ids, dtype=torch.bool) if cfg.engram_layer_ids else None
            out = model(input_ids=ids, ngram_tokens=ngram, engram_mask=mask, use_cache=False)
        logits = torch.nn.functional.linear(out.last_hidden_state.float(), head_w)[0]
    for h in hooks: h.remove()
    o = os.path.join(out_dir, "oracle"); os.makedirs(o, exist_ok=True)
    ids[0].numpy().astype(np.int32).tofile(os.path.join(o, "ids.i32"))
    for f in ("pos3.i32", "is_embed.u8", "embeds.f32"):
        if os.path.exists(os.path.join(o, f)): os.remove(os.path.join(o, f))
    if image:
        pos[:, 0, :].t().contiguous().numpy().astype(np.int32).tofile(os.path.join(o, "pos3.i32"))        # [T, 3]
        is_embed.numpy().astype(np.uint8).tofile(os.path.join(o, "is_embed.u8"))
        feats.numpy().astype(np.float32).tofile(os.path.join(o, "embeds.f32"))
        cj = json.load(open(os.path.join(out_dir, "config.json"))); cj["image_token_id"] = img_id
        json.dump(cj, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    logits.numpy().astype(np.float32).tofile(os.path.join(o, "logits.f32"))
    np.stack([s[0].numpy() for s in streams]).astype(np.float32).tofile(os.path.join(o, "streams.f32"))
    np.stack([r.numpy() for r in routes]).astype(np.int32).tofile(os.path.join(o, "routes.i32"))
    np.stack([m.numpy() for m in margins]).astype(np.float32).tofile(os.path.join(o, "route_margin.f32"))
    (np.stack([r.numpy() for r in recalls]) if recalls else np.zeros((0,), np.int32)).astype(np.int32).tofile(os.path.join(o, "recall.i32"))
    json.dump({"tokens": tokens, "vocab": cfg.vocab_size, "layers": cfg.num_hidden_layers, "hc": cfg.hc_mult,
               "hidden": cfg.hidden_size, "seed": seed, "window": cfg.sliding_window, "group_topk": cfg.compressed_indexer_group_topk,
               "candidate_topk": cfg.candidate_topk, "engram": bool(cfg.engram_layer_ids), "layer_types": cfg.layer_types,
               "mlp_layer_types": cfg.mlp_layer_types, "default_norm_eps": 2.0 ** -7, "norms_pinned": n_eps,
               "recall_layers": recall_layers, "top_k": cfg.num_experts_per_tok}, open(os.path.join(o, "meta.json"), "w"), indent=1)
    mg = torch.stack(margins)
    print(f"oracle: T = {tokens}, vocab {cfg.vocab_size}, window {cfg.sliding_window}, recall budget {cfg.candidate_topk} tokens, "
          f"layers {''.join('I' if t == 'indexed_attention' else 'w' for t in cfg.layer_types)}, indexer ran in layers {recall_layers}")
    print(f"logits: mean {logits.mean():.4f} std {logits.std():.4f}, finite {bool(torch.isfinite(logits).all())}, top-1 of the last row {int(logits[-1].argmax())}")
    print("stream std entering each layer:", " ".join(f"{s.std():.3f}" for s in streams))
    print(f"route margins: median {mg.median():.4f}, {(mg < 1e-3).sum().item()} of {mg.numel()} rows under 0.001, {(mg < 3e-3).sum().item()} under 0.003")

if __name__ == "__main__": main()
