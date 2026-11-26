#!/usr/bin/env python3
"""Merge fine-tunes of one base model by averaging their weight changes (task arithmetic / model soup).

  merge.py --base DIR --out DIR MODEL_DIR[:WEIGHT] [MODEL_DIR[:WEIGHT] ...] [--scale 1.0]

out = base + scale * sum_i w_i (model_i - base) / sum_i w_i, tensor by tensor, for every tensor that differs from the
base in some model (machx-train folds LoRA into the BF16 attention weights only; any other changed tensor must be BF16
or F32 too, or the merge refuses). Unchanged tensors and every other file are the base's. The weight file is a byte
copy of the base's with the changed tensors overwritten in place: same header, offsets and dtypes. CPU only.
"""
import argparse
import json
import os
import shutil
import struct
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("models", nargs="+")
a = ap.parse_args()

WEIGHTS = "model-00001-of-00001.safetensors"


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def as_f32(buf, dtype):
    if dtype == "BF16":
        return (np.frombuffer(buf, np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype == "F32":
        return np.frombuffer(buf, np.float32).copy()
    raise SystemExit(f"merge: a changed tensor has dtype {dtype}; only BF16 / F32 can be averaged here")


def to_bytes(x, dtype):
    if dtype == "F32":
        return x.astype(np.float32).tobytes()
    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16                       # round to nearest even
    return u.astype(np.uint16).tobytes()


models = []
for m in a.models:
    path, w = m.rsplit(":", 1) if ":" in os.path.basename(m) else (m, "")
    models.append((path, float(w) if w else 1.0))
wsum = sum(w for _, w in models)

bh, boff = header(os.path.join(a.base, WEIGHTS))
mh = []
for p, _ in models:
    h, off = header(os.path.join(p, WEIGHTS))
    if {k: (v["dtype"], v["shape"]) for k, v in h.items()} != {k: (v["dtype"], v["shape"]) for k, v in bh.items()}:
        raise SystemExit(f"merge: {p} does not have the base's tensors (names, dtypes, shapes)")
    mh.append((h, off))

if os.path.exists(a.out):
    raise SystemExit(f"merge: {a.out} exists")
tmp = a.out + ".writing"
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)
for name in os.listdir(a.base):
    s, d = os.path.join(a.base, name), os.path.join(tmp, name)
    if name in (WEIGHTS, "TRAINING.md") or name.endswith(".writing"):
        continue
    (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s, d)
print("copying the base weight file ...", flush=True)
shutil.copyfile(os.path.join(a.base, WEIGHTS), os.path.join(tmp, WEIGHTS))

bf = open(os.path.join(a.base, WEIGHTS), "rb")
mfs = [open(os.path.join(p, WEIGHTS), "rb") for p, _ in models]
out = open(os.path.join(tmp, WEIGHTS), "r+b")
changed, total = 0, 0
for name, info in bh.items():
    s, e = info["data_offsets"]
    bf.seek(boff + s)
    braw = bf.read(e - s)
    raws = []
    for f, (h, off) in zip(mfs, mh):
        ms, me = h[name]["data_offsets"]
        f.seek(off + ms)
        raws.append(f.read(me - ms))
    total += 1
    if all(r == braw for r in raws):
        continue
    base = as_f32(braw, info["dtype"])
    delta = sum(w * (as_f32(r, info["dtype"]) - base) for r, (_, w) in zip(raws, models)) / wsum
    out.seek(boff + s)
    out.write(to_bytes(base + a.scale * delta, info["dtype"]))
    changed += 1
out.close()
names = ", ".join(f"{os.path.basename(p.rstrip('/'))} x{w:g}" for p, w in models)
open(os.path.join(tmp, "TRAINING.md"), "w").write(
    f"# Training record\n\n- Merged (average of weight changes, scale {a.scale:g}) from: {names}\n"
    f"- Base: {os.path.basename(a.base.rstrip('/'))}\n- Tensors changed: {changed} of {total}\n")
os.rename(tmp, a.out)
print(f"merged {changed} of {total} tensors -> {a.out}")
