#!/usr/bin/env python3
"""Phase 6a golden: run the reference's NgramHashState + Engram (layer 1) on real weights.

The 91.5 GB embedding table is NOT loaded: ParallelEngramEmbedding is replaced by an mmap-backed
gather with the identical dequant, so only the 24 rows per position are ever touched. Everything
else -- the hash, the wkv Linear, the gate, the residual -- is the shipped code.

  usage: golden_engram.py <out_dir> [model_dir]
"""
import json, os, struct, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
OUT = sys.argv[1]
sys.path.insert(0, HERE)
sys.path.append(os.path.join(D, "inference"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors import safe_open
import kernel as K
import engram as E
import model as ref

cfg = json.load(open(f"{D}/config.json")); tc = cfg["text_config"]
args = ref.ModelArgs(
    dtype="fp8", max_batch_size=1, max_seq_len=4096,
    dim=tc["hidden_size"], hc_mult=tc["hc_mult"], norm_eps=tc["rms_norm_eps"],
    engram_layer_ids=tuple(tc["engram_layer_ids"]),
    engram_num_embeddings=tuple(tc["engram_num_embeddings"]),
    engram_max_ngram_size=tc["engram_max_ngram_size"], engram_vocab_size=tc["engram_vocab_size"],
    engram_n_heads=tc["engram_n_heads"], engram_head_dim=tc["engram_head_dim"],
    engram_pad_id=tc["engram_pad_token_id"],
    engram_compressed_vocab_size=tc["engram_compressed_vocab_size"],
)
ref.default_dtype = torch.float8_e4m3fn
ref.world_size, ref.rank = 1, 0

wm = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
_h = {}
def get(n):
    sh = wm[n]
    if sh not in _h: _h[sh] = safe_open(f"{D}/{sh}", framework="pt")
    return _h[sh].get_tensor(n)
def locate(n):
    """(path, absolute byte offset, shape) of a tensor, for np.memmap."""
    sh = wm[n]; p = f"{D}/{sh}"
    with open(p, "rb") as f:
        hl, = struct.unpack("<Q", f.read(8)); hdr = json.loads(f.read(hl))
    v = hdr[n]; return p, 8 + hl + v["data_offsets"][0], v["shape"]

# ---- the table, mmap'd: the reference's dequant, applied to 24 rows at a time ----------
class MmapEngramEmbedding(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        pw, ow, shw = locate(name + ".weight"); ps, os_, shs = locate(name + ".scale")
        self.w = np.memmap(pw, dtype=np.uint8, mode="r", offset=ow, shape=tuple(shw))
        self.s = np.memmap(ps, dtype=np.uint8, mode="r", offset=os_, shape=tuple(shs))
        self.block = 32
    def forward(self, idx):
        flat = idx.reshape(-1).numpy()
        rows = torch.from_numpy(np.ascontiguousarray(self.w[flat])).view(torch.float8_e4m3fn).float()
        sc   = torch.from_numpy(np.ascontiguousarray(self.s[flat])).view(torch.float8_e8m0fnu).float()
        v = rows.unflatten(-1, (-1, self.block)) * sc.unsqueeze(-1)
        # the reference rounds to bf16 here; keep BOTH so the engine (which keeps fp32) can be
        # compared exactly at the gather and the difference downstream is understood
        v = v.flatten(-2)
        self.last_fp32 = v.view(*idx.shape, -1).clone()
        return v.to(torch.bfloat16).view(*idx.shape, -1)

layout = E.EngramLayout.from_args(args)
tok = AutoTokenizer.from_pretrained(D)
hasher = E.NgramHashState(args, layout, tok)

text = ("The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog "
        "again, and the lazy dog does not care. THE QUICK BROWN FOX -- the quick brown fox -- "
        "jumps. In 2026 the inference engine ran DeepSeek V4.1 on two Arc B70 cards; in 2026 the "
        "inference engine ran it again. Numbers like 12345 and 67890 tokenise in groups of three.")
ids = tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
L = ids.shape[1]
print(f"{L} tokens; first 8 ids {ids[0,:8].tolist()}")
assert L >= 64

with torch.inference_mode():
    hashes = hasher(ids, 0, None)                          # [1, L, 2, 24]
print(f"hash ids {tuple(hashes.shape)}, range [{hashes.min().item():,}, {hashes.max().item():,}]")

# ---- Engram at layer 1 -----------------------------------------------------------------
LAYER = 1
ref.ParallelEngramEmbedding = lambda n, d: MmapEngramEmbedding("layers.1.engram.embed")
eng = ref.Engram(args, LAYER, layout)
def P(t): return torch.nn.Parameter(t, requires_grad=False)
eng.wkv.weight = P(get("layers.1.engram.wkv.weight"))
eng.wkv.weight.scale = eng.wkv.scale = P(get("layers.1.engram.wkv.scale"))
eng.q_weight = P(get("layers.1.engram.q_weight").float())
eng.k_weight = P(get("layers.1.engram.k_weight").float())

g = torch.Generator().manual_seed(20260912)
h = (torch.randn(1, L, args.hc_mult, args.dim, generator=g) * 0.1).float()

with torch.inference_mode():
    hid = hashes[:, :, eng.layer_hash_index, :]
    out = eng(h, hid, None)
    # stage recompute with the module's own pieces
    emb = eng.embed(hid)                                   # bf16 [1, L, 24, 256]
    emb_fp32 = eng.embed.last_fp32
    kv = eng.wkv(emb.flatten(-2))
    key, value = kv.split([args.hc_mult * args.dim, args.dim], dim=-1)
    key = key.float().unflatten(-1, (args.hc_mult, args.dim))
    weight = eng.q_weight * eng.k_weight
    rstd = torch.rsqrt(h.square().mean(-1) + args.norm_eps) * torch.rsqrt(key.square().mean(-1) + args.norm_eps)
    dot = (h * weight * key).sum(-1) * rstd * args.dim ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    out2 = h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)
    assert torch.allclose(out2, out, atol=0, rtol=0), "stage recompute must reproduce Engram.forward"

print(f"dot: min {dot.min():.4f} max {dot.max():.4f}, negatives {(dot<0).sum().item()} of {dot.numel()}, "
      f"|dot|>1e-6: {(dot.abs()>1e-6).sum().item()}")
print(f"gate: min {gate.min():.4f} max {gate.max():.4f}")
print(f"out - h: absmax {(out-h).abs().max():.4f}  (value absmax {value.abs().max():.4f})")

def dump(n, t): t.detach().contiguous().float().numpy().tofile(f"{OUT}/{n}.f32")
ids.to(torch.int32).numpy().tofile(f"{OUT}/eng_ids.i32")
hashes.to(torch.int64).numpy().tofile(f"{OUT}/eng_hashes.i64")
dump("eng_h_in", h); dump("eng_emb_fp32", emb_fp32); dump("eng_kv", kv)
dump("eng_dot", dot); dump("eng_gate", gate); dump("eng_out", out)
json.dump({"L": L, "layer": LAYER, "layer_hash_index": eng.layer_hash_index,
           "n_hash_cols": hid.shape[-1], "dim": args.dim, "hc_mult": args.hc_mult,
           "head_dim": layout.head_dim, "norm_eps": args.norm_eps}, open(f"{OUT}/eng_meta.json", "w"))
print(f"wrote engram goldens to {OUT}")
