#!/usr/bin/env python3
"""Phase 56 golden: the reference's NgramHashState with a token mask (image spans are DEAD tokens).

Writes <out_dir>/engram_mask.bin: magic "DS41EMK1", u32 L, u32 n_layers, u32 n_cols, then i32 ids[L] with every
masked (image) position replaced by a NEGATIVE id -- the engine's encoding of an image position -- and
i64 hashes[n_layers][L][n_cols] from the shipped hasher run with token_mask.
Two sequences back to back: a prefill from position 0, and the SAME sequence hashed as a prefix + a continuation
(start_pos > 0) so the cache's look-back across the split is covered; both must give the same hashes.

  usage: golden_engram_mask.py <out_dir> [model_dir]
"""
import json, os, struct, sys
import numpy as np

D = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
sys.path.append(os.path.join(D, "inference"))
import torch
from transformers import AutoTokenizer
import engram as E
import model as ref

cfg = json.load(open(f"{D}/config.json")); tc = cfg["text_config"]
args = ref.ModelArgs(
    dtype="fp8", max_batch_size=1, max_seq_len=4096,
    dim=tc["hidden_size"], hc_mult=tc["hc_mult"], norm_eps=tc["rms_norm_eps"],
    engram_layer_ids=tuple(tc["engram_layer_ids"]), engram_num_embeddings=tuple(tc["engram_num_embeddings"]),
    engram_max_ngram_size=tc["engram_max_ngram_size"], engram_vocab_size=tc["engram_vocab_size"],
    engram_n_heads=tc["engram_n_heads"], engram_head_dim=tc["engram_head_dim"], engram_pad_id=tc["engram_pad_token_id"],
    engram_compressed_vocab_size=tc["engram_compressed_vocab_size"],
)
layout = E.EngramLayout.from_args(args)
tok = AutoTokenizer.from_pretrained(D)
hasher = E.NgramHashState(args, layout, tok)

text = ("The quick brown fox jumps over the lazy dog. " * 6) + "Describe the picture above, then the one below. " + ("Numbers: 1 2 3 4 5 6 7 8 9 10. " * 4)
ids = torch.tensor(tok.encode(text), dtype=torch.int64)
L = ids.numel()
mask = torch.ones(L, dtype=torch.bool)                 # True = text
for a, b in ((0, 3), (17, 40), (41, 42), (L - 9, L - 4)):   # spans: at the start, a long one, a single slot, near the end
    mask[a:b] = False
img_id = int(json.load(open(f"{D}/inference/config.json"))["image_token_id"])
ref_ids = torch.where(mask, ids, torch.tensor(img_id))  # the reference carries image_token_id at image positions

whole = hasher(ref_ids.unsqueeze(0), 0, mask.unsqueeze(0))[0]                       # [L, n_layers, n_cols]
cut = 37                                                                            # inside the long image span
hasher(ref_ids[:cut].unsqueeze(0), 0, mask[:cut].unsqueeze(0))
cont = hasher(ref_ids[cut:].unsqueeze(0), cut, mask[cut:].unsqueeze(0))[0]
assert torch.equal(whole[cut:], cont), "the reference's own split disagrees with its whole-sequence hash"

eng_ids = ids.clone().to(torch.int32)
eng_ids[~mask] = -1 - torch.arange(int((~mask).sum()), dtype=torch.int32)           # any negative ids
n_layers, n_cols = whole.shape[1], whole.shape[2]
with open(f"{OUT}/engram_mask.bin", "wb") as f:
    f.write(b"DS41EMK1"); f.write(struct.pack("<3I", L, n_layers, n_cols))
    f.write(eng_ids.numpy().astype("<i4").tobytes())
    f.write(whole.permute(1, 0, 2).contiguous().numpy().astype("<i8").tobytes())
print(f"engram_mask.bin: L {L}, {int((~mask).sum())} image positions in 4 spans, {n_layers} engram layers x {n_cols} hash columns; split at {cut} agrees")
