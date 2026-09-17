#!/usr/bin/env python3
"""Precompute the engram hash tables ONCE, using the reference's own functions.

The compressed-token map (tokenizer normalisation), the bucket primes and the per-layer hash
multipliers are functions of the tokenizer and config only. They are produced here by calling
the shipped engram.py -- build_compressed_token_map, EngramLayout.from_args,
compute_hash_multipliers -- not by reimplementing any of them, and serialised for the engine.

  usage: engram_tables.py <out_dir> [model_dir]
Writes <out_dir>/engram_tables.json (small ints) and <out_dir>/engram_token_map.i32.
"""
import json, os, sys
import numpy as np

D = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(D, "inference"))

import torch
from transformers import AutoTokenizer
import engram as E
import model as ref

cfg = json.load(open(f"{D}/config.json"))
tc = cfg["text_config"]
args = ref.ModelArgs(
    engram_layer_ids=tuple(tc["engram_layer_ids"]),
    engram_num_embeddings=tuple(tc["engram_num_embeddings"]),
    engram_max_ngram_size=tc["engram_max_ngram_size"],
    engram_vocab_size=tc["engram_vocab_size"],
    engram_n_heads=tc["engram_n_heads"],
    engram_head_dim=tc["engram_head_dim"],
    engram_pad_id=tc["engram_pad_token_id"],
    engram_compressed_vocab_size=tc["engram_compressed_vocab_size"],
)

tok = AutoTokenizer.from_pretrained(D)
print(f"tokenizer: {len(tok)} ids")
token_map, cvocab = E.build_compressed_token_map(tok)
print(f"compressed vocab: {cvocab}  (config says {args.engram_compressed_vocab_size})")
assert cvocab == args.engram_compressed_vocab_size, "compressed vocab size disagrees with config -> every hash would differ"

layout = E.EngramLayout.from_args(args)
mult = E.compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, cvocab)
flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
offsets = [np.cumsum([0, *sizes[:-1]]).tolist() for sizes in flat]

np.asarray(token_map, dtype=np.int32).tofile(f"{OUT}/engram_token_map.i32")
json.dump({
    "layer_ids": list(layout.layer_ids),
    "max_ngram_size": layout.max_ngram_size,
    "n_heads": layout.n_heads,
    "head_dim": layout.head_dim,
    "num_embeddings": list(layout.num_embeddings),
    "compressed_vocab_size": cvocab,
    "pad_id_compressed": token_map[args.engram_pad_id],
    "primes": [[list(h) for h in layer] for layer in layout.primes],   # [layer][ngram-1][head]
    "offsets": offsets,                                                # [layer][24]
    "multipliers": mult.tolist(),                                      # [layer][max_ngram_size]
    "token_map_len": len(token_map),
}, open(f"{OUT}/engram_tables.json", "w"), indent=1)
print(f"primes[0][0][:3] = {layout.primes[0][0][:3]}  multipliers[0] = {mult[0].tolist()}")
print(f"pad_id {args.engram_pad_id} -> compressed {token_map[args.engram_pad_id]}")
print(f"table rows: layer 1 = {layout.num_embeddings[0]:,}, offsets[0][-1] + prime = "
      f"{offsets[0][-1] + flat[0][-1]:,} (must fit)")
assert offsets[0][-1] + flat[0][-1] <= layout.num_embeddings[0]
print(f"wrote {OUT}/engram_tables.json + engram_token_map.i32")
