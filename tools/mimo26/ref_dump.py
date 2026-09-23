#!/usr/bin/env python3
"""P1 reference dump for the MiMo-V2.6 port (docs/mimo26/00_PORT_PLAN.md, P1 gate (a)/(b)).

Independent reference values the engine's host dequant is compared against, bit for bit:
  * the fused qkv projections of layers 0 (full attention), 1 (SWA) and MTP 0, dequantised by
    llama.cpp's own converter (`conversion/mimo.py::MimoV2Model._tp_aware_qkv_dequant`), which
    de-interleaves the TP=4 row shards and applies the per-rank 128x128 scale blocks;
  * layer 0's dense gate_proj, plain 128x128 block dequant (torch);
  * the first rows of the embeddings / lm_head / final norm (bf16 -> f32);
  * routed experts read from the converted GGUF (Oracle A) through gguf-py's `dequantize`, so the
    MXFP4 bytes went safetensors -> converter -> GGUF -> gguf-py, a path that shares nothing with the
    engine's reader. 8 experts in full, 32 more sampled every `stride` elements.

Run (the venv has torch + safetensors; gguf and the converter come from the llama.cpp checkout):
  PYTHONPATH=~/llama.cpp-mimo:~/llama.cpp-mimo/gguf-py ~/venv/bin/python tools/mimo26/ref_dump.py \
      --model ~/models/MiMo-V2.6-Flash-RL --out results/mimo26/p1/ref \
      [--gguf /path/to/MiMo-V2.6-Flash-RL-BF16-00002-of-00002.gguf]

Output: raw little-endian float32 files + manifest.txt, one entry per line:
  qkv   <L> <file> <N> <K>
  fp8   mlp_gate <L> <file> <N> <K>
  bf16  <embed|lm_head|final_norm> <file> <rows> <cols>
  mxfp4 <L> <E> <gate|up|down> <full|sample> <stride> <file> <N> <K>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open


def load(model_dir, index, name):
    with safe_open(os.path.join(model_dir, index[name]), framework="pt") as f:
        return f.get_tensor(name)


def save(out_dir, name, arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype="<f4"))
    arr.tofile(os.path.join(out_dir, name))
    return name


def dequant_plain(w, s, bn, bk):
    n, k = w.shape
    sf = s.float().repeat_interleave(bn, 0)[:n].repeat_interleave(bk, 1)[:, :k]
    return (w.float() * sf).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf", default=None, help="Oracle A shard holding the blk.* tensors (enables the expert dump)")
    ap.add_argument("--full", type=int, default=8)
    ap.add_argument("--sampled", type=int, default=32)
    ap.add_argument("--stride", type=int, default=97)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    index = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))["weight_map"]
    manifest = []

    from conversion.mimo import MimoV2Model  # llama.cpp's converter, the independent qkv reference

    n_q, hd, vhd = cfg["num_attention_heads"], cfg["head_dim"], cfg["v_head_dim"]
    hybrid = cfg["hybrid_layer_pattern"]
    n_text = cfg["num_hidden_layers"]
    for L, name in [(0, "model.layers.0.self_attn.qkv_proj.weight"),
                    (1, "model.layers.1.self_attn.qkv_proj.weight"),
                    (n_text, "model.mtp.layers.0.self_attn.qkv_proj.weight")]:
        is_swa = True if L >= n_text else hybrid[L] == 1
        n_kv = cfg["swa_num_key_value_heads" if is_swa else "num_key_value_heads"]
        w = load(args.model, index, name)
        s = load(args.model, index, name + "_scale_inv")
        ref = MimoV2Model._tp_aware_qkv_dequant(w, s, n_q, n_kv, hd, vhd).numpy()
        f = save(args.out, f"qkv_L{L}.f32", ref)
        manifest.append(f"qkv {L} {f} {ref.shape[0]} {ref.shape[1]}")
        print("qkv", L, ref.shape, "tp-detected via converter", flush=True)

    bn, bk = cfg["quantization_config"]["weight_block_size"]
    w = load(args.model, index, "model.layers.0.mlp.gate_proj.weight")
    s = load(args.model, index, "model.layers.0.mlp.gate_proj.weight_scale_inv")
    ref = dequant_plain(w, s, bn, bk)
    f = save(args.out, "mlp0_gate.f32", ref)
    manifest.append(f"fp8 mlp_gate 0 {f} {ref.shape[0]} {ref.shape[1]}")
    print("mlp0_gate", ref.shape, flush=True)

    for role, name, rows in [("embed", "model.embed_tokens.weight", 64), ("lm_head", "lm_head.weight", 64),
                             ("final_norm", "model.norm.weight", 1)]:
        t = load(args.model, index, name)
        t = t[:rows] if t.dim() == 2 else t.reshape(1, -1)
        ref = t.float().numpy()
        f = save(args.out, f"{role}.f32", ref)
        manifest.append(f"bf16 {role} {f} {ref.shape[0]} {ref.shape[1]}")

    if args.gguf:
        from gguf import GGUFReader
        from gguf.quants import dequantize

        r = GGUFReader(args.gguf)
        tensors = {t.name: t for t in r.tensors}
        rng = np.random.default_rng(args.seed)
        n_moe_layers = [L for L in range(n_text) if cfg["moe_layer_freq"][L] == 1]
        n_exp = cfg["n_routed_experts"]
        picks = [(1, 0), (n_moe_layers[-1], 255)]   # the first and the last MoE layer always (P1 gate finding 3)
        while len(picks) < args.full + args.sampled:
            p = (int(rng.choice(n_moe_layers)), int(rng.integers(n_exp)))
            if p not in picks:
                picks.append(p)
        for i, (L, E) in enumerate(picks):
            kind = "full" if i < args.full else "sample"
            for proj, gname in [("gate", "ffn_gate_exps"), ("up", "ffn_up_exps"), ("down", "ffn_down_exps")]:
                t = tensors[f"blk.{L}.{gname}.weight"]
                data = np.asarray(t.data)
                if data.ndim != 3 or data.shape[0] != n_exp:
                    sys.exit(f"unexpected GGUF tensor layout for blk.{L}.{gname}: {data.shape} {t.tensor_type}")
                ref = dequantize(data[E], t.tensor_type)   # [N, K] fp32
                N, K = ref.shape
                flat = ref.reshape(-1)
                if kind == "sample":
                    flat = flat[::args.stride]
                f = save(args.out, f"exp_L{L}_E{E}_{proj}.f32", flat)
                manifest.append(f"mxfp4 {L} {E} {proj} {kind} {args.stride if kind == 'sample' else 1} {f} {N} {K}")
            print("expert", L, E, kind, flush=True)

    with open(os.path.join(args.out, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print("wrote", len(manifest), "entries to", args.out)


if __name__ == "__main__":
    main()
