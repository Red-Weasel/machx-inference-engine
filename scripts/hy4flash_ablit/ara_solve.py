#!/usr/bin/env python3
"""Hy4-flash: write maps in closed form (the quadratic form of ara_fit.py's objective; seconds a candidate).

  python ara_solve.py --ie-dir DIR --good w_train_harmless.f32 --bad w_train_harmful.f32 --rows 4 --general w_general.f32
                      --out <prefix> [--lam 10] [--rho 0.01] [--over 0.5] [--knn 8] [--rank 32] [--layers 0-41]

Per write, in u = y / g (unit rms): every bad row u_b gets a target t_b = the mean of its k nearest good rows at the same
position, and the map M (u' = unit(u + M u)) solves
    min  mean_b |M u_b - (1 + over) (t_b - u_b)|^2  +  lam * mean_e |M u_e|^2  +  rho * |M|^2
with e over ordinary text (the general recording) and the good rows: bad writes move onto the good ones (and `over`
beyond), everything else stays. M^T = (Gb + lam Ge + rho I)^-1 Ub^T D / Nb, then its best rank-r form. Output as
ara_fit.py: <prefix>.A.f32 / .B.f32 in y space for the run-time test, <prefix>.u.npz for ara_bake.py.
"""
import argparse, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ara_fit import gains, unit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie-dir", required=True); ap.add_argument("--good", required=True); ap.add_argument("--bad", required=True)
    ap.add_argument("--rows", type=int, default=4); ap.add_argument("--general", required=True); ap.add_argument("--general-rows", type=int, default=16)
    ap.add_argument("--out", required=True); ap.add_argument("--lam", type=float, default=10.0); ap.add_argument("--rho", type=float, default=0.01)
    ap.add_argument("--over", type=float, default=0.5); ap.add_argument("--knn", type=int, default=8); ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--layers", default=""); ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--target", default="mean", choices=["mean", "knn"])
    ap.add_argument("--kinds", default="both", choices=["both", "attn", "ffn"])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    g, nL, H = gains(os.path.expanduser(a.ie_dir)); W = 2 * nL
    yg = torch.from_numpy(np.fromfile(a.good, dtype=np.float32).reshape(-1, W, a.rows, H))
    yb = torch.from_numpy(np.fromfile(a.bad, dtype=np.float32).reshape(-1, W, a.rows, H))
    ge = np.memmap(a.general, dtype=np.float32, mode="r").reshape(-1, W, a.general_rows, H)
    lo, hi = (0, nL - 1) if not a.layers else [int(x) for x in a.layers.split("-")]
    r = a.rank
    Ay = np.zeros((W, r, H), np.float32); By = np.zeros((W, r, H), np.float32); Au = np.zeros((W, r, H), np.float32); Bu = np.zeros((W, r, H), np.float32)
    t0 = time.time(); rep = []
    for w in range(W):
        if w // 2 < lo or w // 2 > hi: continue
        if a.kinds != "both" and (w % 2 == 0) != (a.kinds == "attn"): continue
        gw = g[w]; ok = gw.abs() > 1e-6; gi = torch.where(ok, 1.0 / gw, torch.zeros_like(gw))
        ug = unit(yg[:, w] * gi).double(); ub = unit(yb[:, w] * gi).double()            # [prompts, rows, H]
        x = torch.from_numpy(np.ascontiguousarray(ge[:, w])).reshape(-1, H); x = x[x.abs().sum(-1) > 0]
        E = torch.cat([unit(x * gi).double(), ug.reshape(-1, H)])
        D = torch.empty_like(ub)
        for p in range(a.rows):
            if a.target == "mean":      # the class difference at this position: what a bad prompt adds, nothing of its content
                D[:, p] = (1.0 + a.over) * (ug[:, p].mean(0) - ub[:, p].mean(0))
            else:
                d = torch.cdist(ub[:, p], ug[:, p]); nn = d.topk(a.knn, dim=1, largest=False).indices
                D[:, p] = (1.0 + a.over) * (ug[:, p][nn].mean(1) - ub[:, p])
        Ub = ub.reshape(-1, H); D = D.reshape(-1, H); Nb, Ne = Ub.shape[0], E.shape[0]
        G = Ub.T @ Ub / Nb + a.lam * (E.T @ E) / Ne + a.rho * torch.eye(H, dtype=torch.float64)
        Mt = torch.linalg.solve(G, Ub.T @ D / Nb)                                        # u M^T = the row's change
        P, S, Q = torch.svd_lowrank(Mt, q=min(H, r + 16), niter=4)
        A = (P[:, :r] * S[:r]).T.contiguous().float(); B = Q[:, :r].T.contiguous().float()   # change = (u @ A^T) @ B
        ch = lambda u: float((((u.float() @ A.T) @ B) ** 2).sum(-1).mean().sqrt() / H ** 0.5)
        fit_err = float(((Ub.float() @ A.T) @ B - D.float()).norm() / D.norm())
        rep.append((w, ch(E), ch(Ub), float((D ** 2).sum(-1).mean().sqrt() / H ** 0.5), fit_err))
        Au[w] = A.numpy(); Bu[w] = B.numpy(); Ay[w] = (A * gi).numpy(); By[w] = (B * gw * ok).numpy()
        if w % 12 == 0 or w == W - 1:
            print(f"write {w:2d}: ordinary rows move {rep[-1][1] * 100:5.2f} % rms, bad rows {rep[-1][2] * 100:5.1f} % (asked {rep[-1][3] * 100:5.1f} %, unreached {fit_err * 100:4.0f} %)  [{time.time() - t0:.0f} s]", flush=True)
    Ay.tofile(a.out + ".A.f32"); By.tofile(a.out + ".B.f32")
    np.savez(a.out + ".u.npz", A=Au, B=Bu, rank=r, preserve=a.lam, steer=a.rho, over=a.over, knn=a.knn, layers=np.array([lo, hi]))
    q = np.array(rep)
    print(f"{len(rep)} writes: ordinary rows move {q[:, 1].mean() * 100:.2f} % rms, bad rows {q[:, 2].mean() * 100:.1f} % (asked {q[:, 3].mean() * 100:.1f} %); {time.time() - t0:.0f} s; {a.out}")

if __name__ == "__main__": main()
