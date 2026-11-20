#!/usr/bin/env python3
"""Hy4-flash: make an engine directory independent of the 440 GB download.

  python make_selfcontained.py <download_dir> <ie_dir> [--link-tables]

The engine serves from <ie_dir>: the 8-bit expert pack (experts-q8-LNN.bin), the n-gram tables as exact FP8
(engram-<id>.fp8, written by `ie-hy4flash-run <ie_dir> --pack-tables`) and everything else from a source directory. This
copies that "everything else" -- every shard of the download that holds neither full-precision experts nor an n-gram
table (62 files, about 15.5 GiB: attention, the shared maps, embeddings, vision, the drafter) and the small files --
into <ie_dir>/weights/, writes an index for them, and points the sidecar's source_dir there.
--link-tables also links the four table shards (so the directory works before the FP8 files exist); the links and the
index entries are dropped by running the script again without the flag once the FP8 files are written.
Nothing in <download_dir> is changed.
"""
import argparse, glob, json, os, shutil, struct, sys

def header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; return json.loads(fh.read(n))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("download"); ap.add_argument("ie_dir"); ap.add_argument("--link-tables", action="store_true")
    a = ap.parse_args()
    src = os.path.abspath(a.download); ie = os.path.abspath(os.path.expanduser(a.ie_dir)); dst = os.path.join(ie, "weights")
    os.makedirs(dst, exist_ok=True)
    weight_map = {}; copied = linked = skipped = 0; nbytes = 0
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        h = header(f); names = [k for k in h if k != "__metadata__"]; base = os.path.basename(f)
        is_exp = any(".mlp.experts." in k for k in names)
        is_tab = any(k.endswith("multi_head_embedding.embedding.weight") for k in names)
        if is_exp and any(".mlp.experts." not in k for k in names): raise SystemExit(f"{base} mixes expert tensors with others")
        out = os.path.join(dst, base)
        if is_exp: skipped += 1; continue
        if is_tab:
            if len(names) != 1: raise SystemExit(f"{base}: an n-gram table shares its file with other tensors")
            if os.path.islink(out): os.remove(out)
            if a.link_tables: os.symlink(f, out); linked += 1
            else: continue
        else:
            if not (os.path.isfile(out) and not os.path.islink(out) and os.path.getsize(out) == os.path.getsize(f)):
                shutil.copy2(f, out + ".tmp"); os.replace(out + ".tmp", out)
            copied += 1; nbytes += os.path.getsize(out)
        for k in names: weight_map[k] = base
    json.dump({"metadata": {"format": "pt", "note": "trimmed: no full-precision experts" + ("" if a.link_tables else ", no n-gram tables (engram-<id>.fp8 beside the pack)")},
               "weight_map": weight_map}, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)
    for f in glob.glob(os.path.join(src, "*")):
        b = os.path.basename(f)
        if os.path.isfile(f) and not b.endswith(".safetensors") and b != "model.safetensors.index.json" and not b.startswith("."): shutil.copy2(f, os.path.join(dst, b))
    side_p = os.path.join(ie, "hy4flash_ie.json"); side = json.load(open(side_p))
    if "source_dir_download" not in side: side["source_dir_download"] = side["source_dir"]
    side["source_dir"] = dst; side["expert_format"] = "q8"; side.pop("expert_sqnr_db", None)
    json.dump(side, open(side_p + ".tmp", "w"), indent=1); os.replace(side_p + ".tmp", side_p)
    print(f"{dst}: {copied} shards copied ({nbytes / 2**30:.2f} GiB), {linked} table shards linked, {skipped} expert shards left out, {len(weight_map)} tensors indexed; the sidecar now points here")

if __name__ == "__main__": main()
