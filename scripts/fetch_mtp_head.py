#!/usr/bin/env python3
"""Ranged-fetch the Qwen3.8-Flash-Next MTP head from the HF checkpoint.

Downloads ONLY the mtp.* tensor byte ranges out of each shard (safetensors
header gives exact offsets), avoiding the ~73 GB of full shards. Writes one
consolidated .safetensors at the output path.

usage: fetch_mtp_head.py <out.safetensors>
"""
import json
import os
import struct
import sys
import urllib.request

REPO = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main"


def fetch(url, start=None, end=None, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            if start is not None:
                req.add_header("Range", f"bytes={start}-{end}")
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt+1}: {e}", flush=True)


def shard_header(shard):
    n = struct.unpack("<Q", fetch(f"{REPO}/{shard}", 0, 7))[0]
    hdr = fetch(f"{REPO}/{shard}", 8, 8 + n - 1)
    return json.loads(hdr), 8 + n


def main():
    out_path = sys.argv[1]
    idx = json.loads(fetch(f"{REPO}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    mtp = sorted((k, v) for k, v in wm.items() if k.startswith("mtp."))
    shards = sorted(set(v for _, v in mtp))
    print(f"{len(mtp)} mtp tensors across {len(shards)} shards", flush=True)

    tensors = {}   # name -> (dtype, shape, bytes)
    total = 0
    for si, shard in enumerate(shards):
        hdr, data0 = shard_header(shard)
        for name, s2 in sorted(hdr.items()):
            if name == "__metadata__" or not name.startswith("mtp."):
                continue
            o0, o1 = s2["data_offsets"]
            blob = fetch(f"{REPO}/{shard}", data0 + o0, data0 + o1 - 1)
            tensors[name] = (s2["dtype"], s2["shape"], blob)
            total += len(blob)
            print(f"  [{si+1}/{len(shards)}] {name} {s2['dtype']} {s2['shape']} "
                  f"({len(blob)/2**20:.1f} MiB, total {total/2**30:.2f} GiB)", flush=True)

    # Write a single consolidated safetensors.
    header = {}
    off = 0
    for name, (dt, shp, blob) in sorted(tensors.items()):
        header[name] = {"dtype": dt, "shape": shp, "data_offsets": [off, off + len(blob)]}
        off += len(blob)
    hjson = json.dumps(header).encode()
    pad = (8 - len(hjson) % 8) % 8
    hjson += b" " * pad
    tmp = out_path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for name, (_, _, blob) in sorted(tensors.items()):
            f.write(blob)
    os.replace(tmp, out_path)
    print(f"WROTE {out_path} ({(8+len(hjson)+off)/2**30:.2f} GiB, {len(tensors)} tensors)", flush=True)


if __name__ == "__main__":
    main()
