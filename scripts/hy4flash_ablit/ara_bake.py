#!/usr/bin/env python3
"""Hy4-flash: bake fitted write maps (ara_fit.py) into an engine directory's overlay.

  python ara_bake.py --ie-dir ~/models/Hy4-flash-ie --maps <prefix> --out-dir ~/models/Hy4-flash-ie-abliterated

A write map u' = unit(u + B (A^T u)) on a sublayer's normalised output is the same as replacing the sublayer's last
linear map W by (I + B^T A) W (rows of A, B as stored: [rank, dim]): the norm behind it does the rescaling. Edited:
  attention  self_attn.o_proj.weight
  FFN        mlp.routed_expert_up_proj.weight and mlp.shared_experts.down_proj.weight (their outputs are summed before
             the norm, so both take the same map); layer 0: mlp.down_proj.weight
(a bias of any of them takes the map too). The 512 routed experts of every layer, the n-gram tables and everything else
are the release's: the output directory links them and adds overlay.safetensors (BF16) + abliteration.json.
"""
import argparse, glob, json, os, shutil, struct, time
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def shard_index(src):
    out = {}
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
        for k in h:
            if k != "__metadata__": out[k] = f
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie-dir", required=True); ap.add_argument("--maps", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    ie = os.path.abspath(os.path.expanduser(a.ie_dir)); out = os.path.abspath(os.path.expanduser(a.out_dir))
    side = json.load(open(os.path.join(ie, "hy4flash_ie.json"))); src = side["source_dir"]
    z = np.load(a.maps + ".u.npz"); A = torch.from_numpy(z["A"]).double(); B = torch.from_numpy(z["B"]).double()
    W2, r, H = A.shape; nL = W2 // 2
    idx = shard_index(src); hs = {}
    def get(name):
        f = idx[name]
        if f not in hs: hs[f] = safe_open(f, framework="pt")
        return hs[f].get_tensor(name)
    tensors = {}; rep = []; t0 = time.time()
    def edit(name, w):
        if name not in idx: return
        t = get(name); x = t.double(); col = x.ndim == 1
        if col: x = x.unsqueeze(1)
        new = x + B[w].T @ (A[w] @ x)
        rep.append({"tensor": name, "relative_change": float((new - x).norm() / x.norm())})
        tensors[name] = (new.squeeze(1) if col else new).to(t.dtype).contiguous()
    for L in range(nL):
        p = f"model.layers.{L}."
        if float(B[2 * L].abs().max()) > 0:
            edit(p + "self_attn.o_proj.weight", 2 * L); edit(p + "self_attn.o_proj.bias", 2 * L)
        if float(B[2 * L + 1].abs().max()) > 0:
            for n in ("mlp.routed_expert_up_proj", "mlp.shared_experts.down_proj", "mlp.down_proj"):
                edit(p + n + ".weight", 2 * L + 1); edit(p + n + ".bias", 2 * L + 1)
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(ie, "*")):
        b = os.path.basename(f); dst = os.path.join(out, b)
        if b in ("overlay.safetensors", "abliteration.json"): continue
        if os.path.isdir(f) or os.path.getsize(f) > (64 << 20):          # the packs, the tables, the weights directory: linked
            if not os.path.lexists(dst): os.symlink(f, dst)
        else: shutil.copy2(f, dst)
    save_file(tensors, os.path.join(out, "overlay.safetensors"), metadata={"format": "hy4flash-overlay-1", "method": "write-maps"})
    json.dump({"method": "write maps fitted to recorded sublayer writes (scripts/hy4flash_ablit/ara_fit.py)", "maps": os.path.abspath(a.maps),
               "fit": {k: (z[k].tolist() if z[k].ndim else float(z[k])) for k in ("rank", "preserve", "steer", "over", "knn", "layers")},
               "tensors": len(tensors), "source_dir": src, "report": rep}, open(os.path.join(out, "abliteration.json"), "w"), indent=1)
    ch = np.array([x["relative_change"] for x in rep])
    print(f"overlay: {len(tensors)} tensors, {os.path.getsize(os.path.join(out, 'overlay.safetensors')) / 2**30:.2f} GiB; a tensor changes by {ch.mean() * 100:.2f} % on average "
          f"(largest {ch.max() * 100:.2f} %); {time.time() - t0:.0f} s; {out}")

if __name__ == "__main__": main()
