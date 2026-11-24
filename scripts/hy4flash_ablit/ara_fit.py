#!/usr/bin/env python3
"""Hy4-flash: write maps fitted to the recorded sublayer writes (after Heretic's Arbitrary-Rank Ablation, our own code).

  python ara_fit.py --ie-dir ~/models/Hy4-flash-ie --good w_train_harmless.f32 --bad w_train_harmful.f32 --rows 4
                    --out <prefix> [--rank 16] [--preserve 1.0] [--steer 0.2] [--over 0.5] [--knn 8] [--layers 0-41]

Every sublayer of this model writes y = g (.) u with u = v / rms(v), v = W x the output of its last linear map. The
recordings (ie-hy4flash-run --batch with IE_HY4F_WRITES_OUT) hold y for the last rows of harmless ("good") and harmful
("bad") prompts, [prompts][2 * layers][rows][dim]. Per write a rank-r map u' = unit(u + B (A^T u)) is fitted so that
  the good u change as little as possible,
  the bad u move towards their nearest good ones,
  and (over) away from where the bad ones were;
no refusal direction is assumed. Output <prefix>.A.f32 / .B.f32, [2 * layers][rank][dim], in y space (A / g, B * g):
what ie-hy4flash-run --ablate-list applies at run time, and what bakes into the weights as W' = (I + B A^T) W.
<prefix>.u.npz keeps the u-space pair for the bake. Run with the reference venv (torch on the CPU is enough).
"""
import argparse, glob, json, os, struct, time
import numpy as np
import torch
from safetensors import safe_open

def shard_index(src):
    out = {}
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
        for k in h:
            if k != "__metadata__": out[k] = f
    return out

def gains(ie):
    side = json.load(open(os.path.join(ie, "hy4flash_ie.json"))); src = side["source_dir"]
    cfg = json.load(open(os.path.join(ie, "config.json"))); cfg = cfg.get("text_config", cfg)
    nL, H = cfg["num_hidden_layers"], cfg["hidden_size"]
    idx = shard_index(src); hs = {}
    def get(name):
        f = idx[name]
        if f not in hs: hs[f] = safe_open(f, framework="pt")
        return hs[f].get_tensor(name).float()
    g = torch.zeros(2 * nL, H)
    for L in range(nL):
        p = f"model.layers.{L}."
        g[2 * L] = get(p + "self_attn.post_proj_norm.weight")
        g[2 * L + 1] = get(p + ("mlp.post_fc2_norm.weight" if (p + "mlp.down_proj.weight") in idx else "mlp.combine_output_norm.weight"))
    return g, nL, H

def unit(u): return u * torch.rsqrt((u * u).mean(-1, keepdim=True) + 1e-12)

def knn_mean(a, b, k):
    d = torch.cdist(a, b)
    return d.topk(min(k, b.shape[0]), dim=1, largest=False).values.mean()

def fit(ug, ub, rank, wp, ws, oc, k, seed, gen_u=None, ridge=0.0):
    H = ug.shape[-1]; gen = torch.Generator().manual_seed(seed)
    A = (torch.randn(rank, H, generator=gen) / H ** 0.5).requires_grad_(True)
    B = torch.zeros(rank, H, requires_grad=True)
    opt = torch.optim.LBFGS([A, B], lr=1.0, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    rH = H ** 0.5
    R = ug.shape[1]
    def steer_of(nb):       # per row position: bad rows against the good (and the old bad) rows of the same position
        t = 0.0
        for p in range(R): t = t + knn_mean(nb[:, p], ug[:, p], k) - oc * knn_mean(nb[:, p], ub[:, p], k)
        return t / R / rH
    def loss_fn():
        ng = unit(ug + (ug @ A.T) @ B); nb = unit(ub + (ub @ A.T) @ B)
        keep = ((ng - ug) ** 2).sum(-1).mean() / H * 100.0          # (x100: a 10 % rms change of a good write costs 1)
        if gen_u is not None:                                       # ordinary text at every kind of position: unchanged too
            ne = unit(gen_u + (gen_u @ A.T) @ B)
            keep = 0.5 * keep + 0.5 * ((ne - gen_u) ** 2).sum(-1).mean() / H * 100.0
        steer = steer_of(nb)
        reg = ridge * ((B ** 2).sum() * (A ** 2).sum()) / (rank * rank) if ridge else 0.0
        return wp * keep + ws * steer + reg, keep, steer
    def closure():
        opt.zero_grad(); l, _, _ = loss_fn(); l.backward(); return l
    for _ in range(5): opt.step(closure)
    with torch.no_grad():
        _, keep, _ = loss_fn()
        nb = unit(ub + (ub @ A.T) @ B)
        d = lambda x, y: float(sum(knn_mean(x[:, p], y[:, p], k) for p in range(R)) / R / rH)
        before = d(ub, ug); after = d(nb, ug); floor = d(ug[::2], ug[1::2])
    return A.detach(), B.detach(), float(keep) / 100.0, before, after, floor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie-dir", required=True); ap.add_argument("--good", required=True); ap.add_argument("--bad", required=True)
    ap.add_argument("--rows", type=int, default=4); ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--preserve", type=float, default=1.0)
    ap.add_argument("--steer", type=float, default=0.2); ap.add_argument("--over", type=float, default=0.5); ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--layers", default=""); ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--general", default=""); ap.add_argument("--general-rows", type=int, default=16); ap.add_argument("--general-n", type=int, default=3000)
    ap.add_argument("--ridge", type=float, default=0.0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    g, nL, H = gains(os.path.expanduser(a.ie_dir)); W = 2 * nL
    yg = torch.from_numpy(np.fromfile(a.good, dtype=np.float32).reshape(-1, W, a.rows, H))
    yb = torch.from_numpy(np.fromfile(a.bad, dtype=np.float32).reshape(-1, W, a.rows, H))
    ge = np.memmap(a.general, dtype=np.float32, mode="r").reshape(-1, W, a.general_rows, H) if a.general else None
    lo, hi = 0, nL - 1
    if a.layers: lo, hi = [int(x) for x in a.layers.split("-")]
    Ay = np.zeros((W, a.rank, H), np.float32); By = np.zeros((W, a.rank, H), np.float32)
    Au = np.zeros((W, a.rank, H), np.float32); Bu = np.zeros((W, a.rank, H), np.float32)
    t0 = time.time(); rep = []
    for w in range(W):
        L = w // 2
        if L < lo or L > hi: continue
        gw = g[w]; ok = gw.abs() > 1e-6; gi = torch.where(ok, 1.0 / gw, torch.zeros_like(gw))
        ug = unit(yg[:, w] * gi); ub = unit(yb[:, w] * gi)              # [prompts, rows, H]
        gu = None
        if ge is not None:
            x = torch.from_numpy(np.ascontiguousarray(ge[:, w])).reshape(-1, H)
            x = x[(x.abs().sum(-1) > 0)]                               # (short sequences are zero-padded in front)
            if x.shape[0] > a.general_n: x = x[torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(w))[:a.general_n]]
            gu = unit(x * gi)
        A, B, keep, before, after, floor = fit(ug, ub, a.rank, a.preserve, a.steer, a.over, a.knn, 1000 + w, gu, a.ridge)
        Au[w] = A.numpy(); Bu[w] = B.numpy()
        Ay[w] = (A * gi).numpy(); By[w] = (B * gw * ok).numpy()
        rep.append((w, keep, before, after, floor))
        if w % 12 == 0 or w == W - 1:
            print(f"write {w:2d} (layer {L} {'attn' if w % 2 == 0 else 'ffn '}): good change {keep ** 0.5 * 100:5.1f} % rms; bad -> good distance {before:.3f} -> {after:.3f} (good-good {floor:.3f})  [{time.time() - t0:.0f} s]", flush=True)
    Ay.tofile(a.out + ".A.f32"); By.tofile(a.out + ".B.f32")
    np.savez(a.out + ".u.npz", A=Au, B=Bu, rank=a.rank, preserve=a.preserve, steer=a.steer, over=a.over, knn=a.knn, layers=np.array([lo, hi]))
    r = np.array(rep)
    print(f"{len(rep)} writes: mean good change {np.sqrt(r[:, 1]).mean() * 100:.1f} % rms; bad -> good {r[:, 2].mean():.3f} -> {r[:, 3].mean():.3f} (good-good {r[:, 4].mean():.3f}); {time.time() - t0:.0f} s; {a.out}.{{A,B}}.f32")

if __name__ == "__main__": main()
