#!/usr/bin/env python3
"""Hy4-flash abliteration, step 3: the overlay -- the release's own tensors with one direction removed from what they write.

  python overlay.py --ie-dir ~/models/Hy4-flash-ie --direction <dir>/direction.npy --out-dir ~/models/Hy4-flash-ie-abliterated
                    [--strength 1.0] [--writers attn,moe,dense,ngram,embed] [--layers 0-41]
  python overlay.py ... --basis set.f32 --k K ...       a SET of directions: raw fp32 [K, dim] orthonormal rows, the file
                                                        make_candidates.py writes and ie-hy4flash-run --ablate-list tests
(One direction did not remove this model's refusals, 2026-11-18: 20 per-stream directions over five layers were the first
set that did. With a set R every "u" below is an orthonormal basis of the columns g (.) r_j.)

Every sublayer of this model RMS-normalises its output with a learned gain g before adding it to the residual streams:
write = post_i * g (.) v / rms(v). The write has no component along the direction r exactly when v is orthogonal to
g (.) r, so only the LAST linear map that produces v is edited, W' = (I - s * u u^T) W with u = normalise(g (.) r):
  attn    self_attn.o_proj.weight                                   g = self_attn.post_proj_norm.weight
  moe     mlp.routed_expert_up_proj.weight, mlp.shared_experts.down_proj.weight   g = mlp.combine_output_norm.weight
  dense   mlp.down_proj.weight (layer 0)                             g = mlp.post_fc2_norm.weight
  ngram   engram_layer.value_proj.weight + .bias, against the five vectors pre_gamma (.) r and scale_i (.) pre_gamma (.) r
  embed   embed_tokens.weight rows, against r itself
The 512 routed experts of every layer are NOT touched (the 4-bit pack is the same before and after). The output directory
is an engine directory: the sidecar, config and tokenizer copied, the expert pack linked, overlay.safetensors (BF16) and
abliteration.json. Run with the reference venv.
"""
import argparse, glob, json, os, shutil, struct, sys, time
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
    ap.add_argument("--ie-dir", required=True); ap.add_argument("--direction"); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--basis"); ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--strength", type=float, default=1.0); ap.add_argument("--writers", default="attn,moe,dense,ngram,embed")
    ap.add_argument("--layers", default="")
    a = ap.parse_args()
    ie = os.path.expanduser(a.ie_dir); out = os.path.expanduser(a.out_dir)
    side = json.load(open(os.path.join(ie, "hy4flash_ie.json"))); src = side["source_dir"]
    cfg = json.load(open(os.path.join(ie, "config.json")))["text_config"]
    nL, H, hc = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["hc_mult"]
    writers = set(a.writers.split(","))
    layers = set(range(nL))
    if a.layers:
        layers = set()
        for part in a.layers.split(","):
            if "-" in part: x, y = part.split("-"); layers.update(range(int(x), int(y) + 1))
            else: layers.add(int(part))
    if a.basis:
        R = torch.from_numpy(np.fromfile(a.basis, dtype=np.float32).reshape(a.k, H).astype(np.float64)).T.contiguous()   # [H, k]
        R, _ = torch.linalg.qr(R)                                                    # (already orthonormal; exact in fp64)
    else:
        if not a.direction: raise SystemExit("--direction or --basis")
        r = torch.from_numpy(np.load(a.direction).astype(np.float64)); assert r.numel() == H, (r.numel(), H)
        R = (r / r.norm()).unsqueeze(1)
    K = R.shape[1]
    idx = shard_index(src); handles = {}
    def get(name):
        f = idx[name]
        if f not in handles: handles[f] = safe_open(f, framework="pt")
        return handles[f].get_tensor(name)
    s = a.strength; tensors = {}; report = []
    def project(name, basis):       # W' = W - s * B (B^T W), B orthonormal columns [H, k]
        w = get(name); dt = w.dtype; w64 = w.double()
        comp = basis.T @ (w64 if w64.ndim == 2 else w64.unsqueeze(1))
        new = (w64 if w64.ndim == 2 else w64.unsqueeze(1)) - s * (basis @ comp)
        new = new if w64.ndim == 2 else new.squeeze(1)
        before = float(comp.norm() / w64.norm()); after = float((basis.T @ (new if new.ndim == 2 else new.unsqueeze(1))).norm() / new.norm())
        tensors[name] = new.to(dt).contiguous(); report.append({"tensor": name, "share_along_before": before, "share_along_after_fp64": after})
    def unit(g): q, _ = torch.linalg.qr(g.unsqueeze(1) * R); return q      # an orthonormal basis of {g (.) r_j}
    t0 = time.time()
    for L in sorted(layers):
        p = f"model.layers.{L}."
        if "attn" in writers:
            g = get(p + "self_attn.post_proj_norm.weight").double()
            project(p + "self_attn.o_proj.weight", unit(g))
        dense_layer = (p + "mlp.down_proj.weight") in idx
        if dense_layer and "dense" in writers:
            g = get(p + "mlp.post_fc2_norm.weight").double()
            project(p + "mlp.down_proj.weight", unit(g))
        if not dense_layer and "moe" in writers:
            g = get(p + "mlp.combine_output_norm.weight").double(); u = unit(g)
            project(p + "mlp.routed_expert_up_proj.weight", u)
            project(p + "mlp.shared_experts.down_proj.weight", u)
        if (p + "engram_layer.value_proj.weight") in idx and "ngram" in writers:
            pg = get(p + "engram_layer.unified_pre_gamma").double()
            vecs = [pg.unsqueeze(1) * R] + [(get(p + f"engram_layer.unified_per_hc_scales.{i}").double() * pg).unsqueeze(1) * R for i in range(hc)]
            q, _ = torch.linalg.qr(torch.cat(vecs, dim=1))              # orthonormal basis of the (hc + 1) * K vectors
            project(p + "engram_layer.value_proj.weight", q)
            project(p + "engram_layer.value_proj.bias", q)
        print(f"layer {L}: done ({time.time() - t0:.0f} s)", flush=True)
    if "embed" in writers:
        name = "model.embed_tokens.weight"; e = get(name); dt = e.dtype; e64 = e.double()
        comp = e64 @ R
        new = e64 - s * comp @ R.T
        tensors[name] = new.to(dt).contiguous()
        report.append({"tensor": name, "share_along_before": float(comp.norm() / e64.norm()), "share_along_after_fp64": float((new @ R).norm() / new.norm())})
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(ie, "*")):
        b = os.path.basename(f); dst = os.path.join(out, b)
        if b.startswith("experts-") or b == "model.safetensors.index.json":       # (the 8-bit pack is 147 GiB: linked, never copied)
            if not os.path.lexists(dst): os.symlink(f, dst)                 # the pack is shared, not copied
        elif os.path.isfile(f) and b not in ("overlay.safetensors", "abliteration.json", "pack.log"): shutil.copy2(f, dst)
    save_file(tensors, os.path.join(out, "overlay.safetensors"), metadata={"format": "hy4flash-overlay-1", "strength": str(s)})
    R.T.contiguous().to(torch.float32).numpy().tofile(os.path.join(out, "direction.f32"))   # raw [K, dim] fp32 (K = 1: ie-hy4flash-mini --along)
    n_params = sum(t.numel() for t in tensors.values())
    json.dump({"direction": os.path.abspath(a.basis or a.direction), "directions": K, "strength": s, "writers": sorted(writers), "layers": sorted(layers), "tensors": len(tensors),
               "parameters": n_params, "source_dir": src, "report": report}, open(os.path.join(out, "abliteration.json"), "w"), indent=1)
    mean_before = float(np.mean([x["share_along_before"] for x in report]))
    print(f"overlay: {len(tensors)} tensors, {n_params / 1e9:.2f} B parameters, {os.path.getsize(os.path.join(out, 'overlay.safetensors')) / 2**30:.2f} GiB; "
          f"mean share of a tensor's output along the direction before {mean_before:.4f}; written to {out}")

if __name__ == "__main__": main()
