#!/usr/bin/env python3
"""Bake an Inkling 2 refusal set (uniform per-kind strengths) into the checkpoint's weights: one standalone model folder.

  bake.py --src <abliterated dir> --set ie_ablation_v3.json --out <new dir>

The engine removes s * (r.y) r from a write y AFTER the write's per-channel filter (sconv, whose current-token tap D
dominates). In the weights that is y = D x -> P D x = D M x with M = I - s u v^T, v = D r, u = D^-1 r (Tikhonov-regularised
where a channel's tap is near zero). M is applied to: the embedding table (D = embed_norm), every attention output
projection wo_ud, every expert's w2. The three loop passes share weights, as their strengths are equal in the set.
Experts are re-encoded in their own format (NVFP4 / MXFP8): the block scales stay unless a value no longer fits, and
after nearest rounding the rows with the largest |v| are nudged one code so that v.column keeps its target value (plain
rounding of a low-rank edit to 4 bits would lose most of it). Header, offsets, dtypes and every other tensor are unchanged.
"""
import argparse, json, mmap, os, shutil, struct, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--set", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--reg", type=float, default=0.1, help="regulariser: fraction of the rms tap")
ap.add_argument("--shape-rows", type=int, default=400)
a = ap.parse_args()
FN = "model-00001-of-00001.safetensors"
t0 = time.time()
def log(*x): print(f"{time.time() - t0:7.1f}s", *x, flush=True)

S = json.load(open(os.path.join(a.src, a.set)))
assert S["rows"] == 1 and S["hidden"] == 2304
r = np.fromfile(os.path.join(a.src, S["file"]), dtype=np.float32).astype(np.float64); r /= np.linalg.norm(r)
sa = {c: g["strength"] for g in S["groups"] for c in g.get("attn", [])}
sm = {c: g["strength"] for g in S["groups"] for c in g.get("moe", [])}
se = next(g["strength"] for g in S["groups"] if g.get("embed"))
assert len(sa) == 48 and len(sm) == 48 and len(set(sa.values())) == 1 and len(set(sm.values())) == 1, "strengths are not uniform"
SA, SM = sa[0], sm[0]
log(f"strengths: embedding {se}, attention {SA}, experts {SM}")

os.makedirs(a.out + ".writing")
W = a.out + ".writing"
for f in sorted(os.listdir(a.src)):
    p = os.path.join(a.src, f)
    if f.startswith("ie_") or not os.path.isfile(p) and f != "tiktoken": continue
    if f == "README.md": continue
    if os.path.isdir(p): shutil.copytree(p, os.path.join(W, f))
    else: shutil.copyfile(p, os.path.join(W, f))
log("copied the model files")

fd = open(os.path.join(W, FN), "r+b"); n = struct.unpack("<Q", fd.read(8))[0]; H = json.loads(fd.read(n)); base = 8 + n
mm = mmap.mmap(fd.fileno(), 0)
def view(k, dt):
    b, e = H[k]["data_offsets"]
    return np.frombuffer(mm, dtype=dt, count=(e - b) // np.dtype(dt).itemsize, offset=base + b).reshape(H[k]["shape"])
def bf16_read(k): return (view(k, np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
def bf16_write(k, x):
    u = torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16).contiguous()
    b, e = H[k]["data_offsets"]
    raw = bytes(u.untyped_storage())
    assert len(raw) == e - b, (k, len(raw), e - b)
    mm[base + b:base + e] = raw

def uv(d):
    lam = (a.reg * np.sqrt(np.mean(d ** 2))) ** 2
    return d * r / (d ** 2 + lam), d * r

# ---------------------------------------------------------------------------------------------- embedding, wo
u, v = uv(bf16_read("model.llm.embed_norm.weight"))
E = bf16_read("model.llm.embed.weight")
rho0 = v @ E[:4096].T
for i in range(0, E.shape[0], 8192):
    E[i:i + 8192] -= se * np.outer(u, v @ E[i:i + 8192].T).T
bf16_write("model.llm.embed.weight", E)
rho1 = v @ bf16_read("model.llm.embed.weight")[:4096].T
log(f"embedding: along r after / before = {np.dot(rho1, rho0) / np.dot(rho0, rho0):+.3f} (target {1 - se:+.3f})"); del E

for L in range(24):
    k = f"model.llm.layers.{L}.attn.wo_ud.weight"
    D = bf16_read(f"model.llm.layers.{L}.attn_sconv.weight")[:, 0, 3]
    u, v = uv(D); w = bf16_read(k); t = v @ w
    bf16_write(k, w - SA * np.outer(u, t)); t2 = v @ bf16_read(k)
    print(f"  L{L} wo: {np.dot(t2, t) / np.dot(t, t):+.3f} (target {1 - SA:+.3f}); filter fit {np.linalg.norm(D * u - r):.3f}", flush=True)
log("attention output projections done")

# ---------------------------------------------------------------------------------------------- experts
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
G = np.array([-6, -4, -3, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 3, 4, 6.])
G2N = np.array([15, 14, 13, 12, 11, 10, 9, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
def e4m3():
    o = np.zeros(256)
    for b in range(256):
        s, x, m = b >> 7, (b >> 3) & 15, b & 7
        o[b] = (-1) ** s * (m / 8 * 2.0 ** -6 if x == 0 else (1 + m / 8) * 2.0 ** (x - 7))
    return o
F8 = e4m3()
# float8_e4m3fn: 0x7F and 0xFF are NaN (the largest finite value is 448 = 0x7E); they must never be written
FINITE = np.array([b for b in range(256) if b not in (0x7F, 0xFF)])
pos = FINITE[F8[FINITE] > 0]; pos = pos[np.argsort(F8[pos])]; POSV = F8[pos]

def nearest(sorted_v, x):
    j = np.clip(np.searchsorted(sorted_v, x), 1, len(sorted_v) - 1)
    return j - (x - sorted_v[j - 1] < sorted_v[j] - x)

def bake_nvfp4(L, u, v):
    p = f"model.llm.layers.{L}.mlp.experts.w2_weight"
    C, B = view(p, np.uint8), view(p + "_scale", np.uint8)
    f = view(p + "_amax", np.float32).astype(np.float64) / float(view(p + "_outer_scale_denominator", np.float32)[0])
    top = np.argsort(-np.abs(v))[:a.shape_rows]; num = den = 0.0; moved = 0
    for e in range(128):
        c = C[e]; b = B[e].copy()
        vals = np.empty((2304, 1152)); vals[:, ::2] = E2M1[c & 15]; vals[:, 1::2] = E2M1[c >> 4]
        w = vals * np.repeat(F8[b], 16, axis=1) * f[e]
        t = v @ w; target = t * (1 - SM); w2 = w - SM * np.outer(u, t)
        need = np.abs(w2).reshape(2304, 72, 16).max(-1) / (6 * f[e])
        over = need > F8[b]
        if over.any():
            b[over] = pos[np.clip(np.searchsorted(POSV, need[over]), 0, len(pos) - 1)]
        sc = np.repeat(F8[b], 16, axis=1) * f[e]
        q = np.divide(w2, sc, out=np.zeros_like(w2), where=sc != 0)
        ix = nearest(G, q)
        err = target - v @ (G[ix] * sc)
        for i in top:                                   # one code at most, only where it brings v.column closer
            d = np.sign(err * v[i]).astype(np.int64)
            nw = np.clip(ix[i] + d, 0, 14)
            delta = v[i] * (G[nw] - G[ix[i]]) * sc[i]
            ok = np.abs(err - delta) < np.abs(err)
            ix[i, ok] = nw[ok]; err[ok] -= delta[ok]
        n2 = G2N[ix]
        new = n2[:, ::2] | (n2[:, 1::2] << 4)
        moved += int((new != c).sum()); C[e] = new; B[e] = b
        got = v @ (G[ix] * sc); num += float(got @ t); den += float(t @ t)
    return num / den, moved / C.size

def bake_mxfp8(L, u, v):
    p = f"model.llm.layers.{L}.mlp.experts.w2_weight"
    C, B = view(p, np.uint8), view(p + "_scale", np.uint8)
    o = FINITE[np.argsort(F8[FINITE], kind="stable")]; sv = F8[o]; num = den = 0.0; moved = 0
    for e in range(128):
        c = C[e]; b = B[e].copy()
        w = F8[c] * np.repeat(2.0 ** (b.astype(np.float64) - 127), 32, axis=1)
        t = v @ w; w2 = w - SM * np.outer(u, t)
        need = np.abs(w2).reshape(2304, 36, 32).max(-1) / 448
        cur = 2.0 ** (b.astype(np.float64) - 127)
        if (need > cur).any():
            up = np.ceil(np.log2(np.maximum(need, 1e-300))) + 127
            b = np.where(need > cur, up, b).astype(np.uint8)
        sc = np.repeat(2.0 ** (b.astype(np.float64) - 127), 32, axis=1)
        new = o[nearest(sv, w2 / sc)].astype(np.uint8)
        moved += int((new != c).sum()); C[e] = new; B[e] = b
        got = v @ (F8[new] * sc); num += float(got @ t); den += float(t @ t)
    return num / den, moved / C.size

for L in range(24):
    D = bf16_read(f"model.llm.layers.{L}.mlp_sconv.weight")[:, 0, 3]
    u, v = uv(D)
    nv = (f"model.llm.layers.{L}.mlp.experts.w2_weight_amax") in H
    rho, ch = (bake_nvfp4 if nv else bake_mxfp8)(L, u, v)
    log(f"L{L} experts ({'NVFP4' if nv else 'MXFP8'}): along r after / before = {rho:+.3f} (target {1 - SM:+.3f}); codes changed {100 * ch:.1f} %; filter fit {np.linalg.norm(D * u - r):.3f}")

mm.flush(); mm.close(); fd.close()
json.dump({"baked_from": a.set, "strengths": {"embedding": se, "attention": SA, "experts": SM}, "regulariser": a.reg},
          open(os.path.join(W, "abliteration_bake.json"), "w"), indent=1)
os.rename(W, a.out)
log("DONE", a.out)
