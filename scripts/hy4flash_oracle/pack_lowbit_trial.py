#!/usr/bin/env python3
"""Hy4-flash: what would 4- or 5-bit experts cost in quality? -- measured through the engine's existing 8-bit path.

  python pack_lowbit_trial.py <source_dir> <ie_dir> <trial_dir> --layers 5,20,35 --bits 4 --method gptq|plain
                              [--down-bits B] [--cov hy4_cov.npz | --cov-engine PREFIX] [--damp 0.01] [--chunk 32] [--threads 6]

A symmetric b-bit integer with one fp16 scale per 32 weights (w = d * q, |q| <= 2^(b-1) - 1) is a special case of the 8-bit
slot layout (Hy4fQ8Store: int8 q, fp16 d per 32). So a trial format needs no kernel: the chosen layers' experts are
rounded to b bits and written as ordinary experts-q8-LNN.bin files into <trial_dir>, every other file of <ie_dir> is
linked, and the engine runs <trial_dir> as usual. The arithmetic is exactly what a real b-bit kernel would compute; only
the speed of the smaller format is not there. Measure with
  ie-hy4flash-mini <trial_dir> --oracle <reference> --q8 --force      (the reference's expert picks: the rounding alone)
and scale by (41 / layers converted).

  plain   each 32 weights: the scale (a fraction of max|w| / qmax, the fraction with the least squared error) and
          round-to-nearest
  gptq    the same scales, but the columns are rounded one after another and each rounding error is pushed onto the columns
          not yet rounded, weighted by the covariance of the experts' real inputs (--cov: 'xtx' [41, latent, latent] and
          'N', recorded by the engine) -- the error that matters is the OUTPUT's, not the weight's. gate and up share the
          latent input and its covariance; down's input (an expert's own hidden activation) has no recorded statistics
          and is rounded plainly (--down-bits gives it its own width, e.g. 4-bit gate / up with an 8-bit down).
          With --cov-engine (the engine's recorder, ie-hy4flash-run --cov-out: PREFIX.xtx.f64, PREFIX.hth.f64, PREFIX.json)
          down is rounded with GPTQ too, under the layer's pooled covariance of the experts' hidden activations.
Run with the reference venv.
"""
import argparse, glob, json, os, struct, sys, time
import numpy as np
import torch
from safetensors import safe_open

CLIPS = (1.0, 0.94, 0.88, 0.82, 0.76, 0.70, 0.64)

def shard_index(src):
    out = {}
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
        for k in h:
            if k != "__metadata__": out[k] = f
    return out

def scales(w, qmax):
    """w [R, K] fp32 -> d [R, K / 32] fp32 holding fp16-representable scales with the least squared rounding error"""
    R, K = w.shape
    b = w.reshape(R, K // 32, 32); amax = b.abs().amax(-1)
    best = None; best_err = None
    for c in CLIPS:
        d = (amax * (c / qmax)).to(torch.float16).float()
        d = torch.where(d > 0, d, torch.full_like(d, 1e-8))
        q = torch.round(b / d.unsqueeze(-1)).clamp_(-qmax, qmax)
        err = ((b - q * d.unsqueeze(-1)) ** 2).sum(-1)
        if best is None: best, best_err = d, err
        else:
            take = err < best_err
            best = torch.where(take, d, best); best_err = torch.where(take, err, best_err)
    return best

def round_plain(w, d, qmax):
    R, K = w.shape
    return torch.round(w.reshape(R, K // 32, 32) / d.unsqueeze(-1)).clamp_(-qmax, qmax).reshape(R, K)

def round_gptq(w, d, qmax, U, block=128):
    """Columns in order; the error of column j, divided by U[j, j], is taken off the later columns along U[j, j + 1:]
    (U = the upper Cholesky factor of the inverse input covariance). Returns the integers [R, K]."""
    R, K = w.shape
    W = w.clone(); Q = torch.empty_like(W)
    dcol = d.repeat_interleave(32, dim=1)                    # [R, K]
    for i0 in range(0, K, block):
        i1 = min(K, i0 + block)
        Wb = W[:, i0:i1].clone(); Eb = torch.zeros_like(Wb); Ub = U[i0:i1, i0:i1]
        for j in range(i1 - i0):
            dj = dcol[:, i0 + j]
            q = torch.round(Wb[:, j] / dj).clamp_(-qmax, qmax)
            Q[:, i0 + j] = q
            err = (Wb[:, j] - q * dj) / Ub[j, j]
            if j + 1 < i1 - i0: Wb[:, j + 1:] -= err.unsqueeze(1) * Ub[j, j + 1:].unsqueeze(0)
            Eb[:, j] = err
        if i1 < K: W[:, i1:] -= Eb @ U[i0:i1, i1:]
    return Q

def chol_inv_upper(xtx, n, damp):
    H = torch.from_numpy(np.asarray(xtx, dtype=np.float64)) / float(n)
    dead = torch.diag(H) <= 0
    H[dead, dead] = 1.0
    H += torch.eye(H.shape[0], dtype=torch.float64) * (damp * torch.diag(H).mean())
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    return torch.linalg.cholesky(Hinv, upper=True).float()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("ie_dir"); ap.add_argument("trial_dir")
    ap.add_argument("--layers", required=True); ap.add_argument("--bits", type=int, required=True); ap.add_argument("--method", choices=("plain", "gptq"), required=True)
    ap.add_argument("--down-bits", type=int, default=0)
    ap.add_argument("--down-plain", action="store_true")      # round down plainly even when the hidden-activation statistics exist
    ap.add_argument("--from-q8", action="store_true")       # the source weights = <ie_dir>'s 8-bit pack, dequantized (no BF16 source)
    ap.add_argument("--cov"); ap.add_argument("--cov-engine"); ap.add_argument("--damp", type=float, default=0.01); ap.add_argument("--chunk", type=int, default=32); ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    src, ie, out = os.path.abspath(a.src), os.path.abspath(os.path.expanduser(a.ie_dir)), os.path.abspath(os.path.expanduser(a.trial_dir))
    cfg = json.load(open(os.path.join(src, "config.json"))); cfg = cfg.get("text_config", cfg)
    nL, E, MI, LAT = cfg["num_hidden_layers"], cfg["n_routed_experts"], cfg["moe_intermediate_size"], cfg["routed_expert_hidden_size"]
    moe = [L for L in range(nL) if cfg["mlp_layer_types"][L] == "sparse"]
    layers = [int(x) for x in a.layers.split(",")]
    assert all(L in moe for L in layers), "a listed layer has no routed experts"
    qmax = (1 << (a.bits - 1)) - 1
    qmax_dn = (1 << ((a.down_bits or a.bits) - 1)) - 1
    cov = None; eng = None
    if a.method == "gptq":
        if a.cov_engine:
            meta = json.load(open(a.cov_engine + ".json")); n_moe = len(meta["tokens"])
            eng = {"xtx": np.memmap(a.cov_engine + ".xtx.f64", dtype=np.float64, mode="r").reshape(n_moe, meta["latent"], meta["latent"]),
                   "hth": (np.memmap(a.cov_engine + ".hth.f64", dtype=np.float64, mode="r").reshape(n_moe, meta["moe_inter"], meta["moe_inter"])
                           if os.path.exists(a.cov_engine + ".hth.f64") and sum(meta["jobs"]) > 0 and not a.down_plain else None),
                   "tokens": meta["tokens"], "jobs": meta["jobs"]}
        elif a.cov: cov = np.load(a.cov)
        else: raise SystemExit("--cov or --cov-engine is needed for gptq")
    gq, dq, gd, dd = MI * LAT, LAT * MI, MI * (LAT // 32) * 2, LAT * (MI // 32) * 2
    slot = (2 * gq + dq + 2 * gd + dd + 63) // 64 * 64
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(ie, "*")):                # everything else is the serving directory's own
        b = os.path.basename(f); dst = os.path.join(out, b)
        if os.path.lexists(dst): os.remove(dst)
        if not any(b == f"experts-q8-L{L:02d}.bin" for L in layers) and b != "overlay.safetensors": os.symlink(f, dst)
    idx = {} if a.from_q8 else shard_index(src); report = []
    bits16 = lambda d: d.to(torch.float16).view(torch.int16)
    for L in layers:
        t0 = time.time()
        U = Ud = None; li = moe.index(L)
        if a.method == "gptq" and eng is not None:
            U = chol_inv_upper(eng["xtx"][li], eng["tokens"][li], a.damp)
            Ud = chol_inv_upper(eng["hth"][li], eng["jobs"][li], a.damp) if eng["hth"] is not None else None
        elif a.method == "gptq": U = chol_inv_upper(cov["xtx"][li], cov["N"], a.damp)
        gu, dn = f"model.layers.{L}.mlp.experts.gate_up_proj", f"model.layers.{L}.mlp.experts.down_proj"
        path = os.path.join(out, f"experts-q8-L{L:02d}.bin")
        se_w = 0.0; sw = 0.0                                   # weight error, all three matrices
        import contextlib
        if a.from_q8:                                          # the 8-bit slots: q gate, q up, q down, d gate, d up, d down (fp16 per 32)
            q8 = np.memmap(os.path.join(ie, f"experts-q8-L{L:02d}.bin"), dtype=np.uint8, mode="r").reshape(E, slot)
            def src_chunk(e0, e1):
                n = e1 - e0; b = np.asarray(q8[e0:e1]); o = 2 * gq + dq
                qgu = torch.from_numpy(b[:, :2 * gq].view(np.int8).reshape(n * 2 * MI, LAT).astype(np.float32))
                qdn = torch.from_numpy(b[:, 2 * gq:o].view(np.int8).reshape(n * LAT, MI).astype(np.float32))
                dgu = torch.from_numpy(b[:, o:o + 2 * gd].copy().view(np.float16).reshape(n * 2 * MI, LAT // 32).astype(np.float32))
                ddn = torch.from_numpy(b[:, o + 2 * gd:o + 2 * gd + dd].copy().view(np.float16).reshape(n * LAT, MI // 32).astype(np.float32))
                return qgu * dgu.repeat_interleave(32, 1), qdn * ddn.repeat_interleave(32, 1)
            cm = contextlib.nullcontext((None, None))
        else:
            cm = contextlib.nullcontext((safe_open(idx[gu], framework="pt"), safe_open(idx[dn], framework="pt")))
        with cm as (fg, fd), open(path + ".tmp", "wb") as fo:
            sg, sd = (fg.get_slice(gu), fd.get_slice(dn)) if fg is not None else (None, None)
            for e0 in range(0, E, a.chunk):
                e1 = min(E, e0 + a.chunk); n = e1 - e0
                if a.from_q8: wg, wd = src_chunk(e0, e1)
                else:
                    wg = sg[e0:e1].float().reshape(n * 2 * MI, LAT)       # gate rows then up rows of each expert
                    wd = sd[e0:e1].float().reshape(n * LAT, MI)
                dg = scales(wg, qmax); ddn = scales(wd, qmax_dn)
                qg = round_gptq(wg, dg, qmax, U) if a.method == "gptq" else round_plain(wg, dg, qmax)
                qd = round_gptq(wd, ddn, qmax_dn, Ud) if Ud is not None else round_plain(wd, ddn, qmax_dn)
                se_w += float(((wg - qg * dg.repeat_interleave(32, 1)) ** 2).sum() + ((wd - qd * ddn.repeat_interleave(32, 1)) ** 2).sum())
                sw += float((wg ** 2).sum() + (wd ** 2).sum())
                qg8 = qg.to(torch.int8).reshape(n, 2 * MI, LAT); bg = bits16(dg).reshape(n, 2 * MI, LAT // 32)
                qd8 = qd.to(torch.int8).reshape(n, LAT, MI); bd = bits16(ddn).reshape(n, LAT, MI // 32)
                buf = np.zeros((n, slot), dtype=np.uint8)
                buf[:, 0:gq] = qg8[:, :MI].reshape(n, gq).numpy().view(np.uint8)
                buf[:, gq:2 * gq] = qg8[:, MI:].reshape(n, gq).numpy().view(np.uint8)
                buf[:, 2 * gq:2 * gq + dq] = qd8.reshape(n, dq).numpy().view(np.uint8)
                o = 2 * gq + dq
                buf[:, o:o + gd] = bg[:, :MI].reshape(n, gd // 2).numpy().view(np.uint8).reshape(n, gd)
                buf[:, o + gd:o + 2 * gd] = bg[:, MI:].reshape(n, gd // 2).numpy().view(np.uint8).reshape(n, gd)
                buf[:, o + 2 * gd:o + 2 * gd + dd] = bd.reshape(n, dd // 2).numpy().view(np.uint8).reshape(n, dd)
                fo.write(buf.tobytes())
        os.replace(path + ".tmp", path)
        rel = (se_w / sw) ** 0.5
        report.append({"layer": L, "weight_error": rel})
        print(f"layer {L}: {a.bits}-bit {a.method}, relative weight error {100 * rel:.2f} %, {time.time() - t0:.0f} s", flush=True)
    json.dump({"bits": a.bits, "down_bits": a.down_bits or a.bits, "method": a.method, "layers": layers, "damp": a.damp, "report": report}, open(os.path.join(out, "lowbit_trial.json"), "w"), indent=1)
    print(f"trial directory {out}: {len(layers)} of {len(moe)} expert layers at {a.bits} bits ({a.method}); the rest linked from {ie}")

if __name__ == "__main__": main()
