#!/usr/bin/env python3
"""Delta-Forest probe: how block-sparse is the INT8 delta between a token's
layer-input activation and the most similar of its recent predecessors?

Offline numpy analysis of the activation dump written by ie-ds4-ppl
(DS4_ACT_DUMP=<file>).  Dump format: 4 x uint32 LE header
[magic 0x31544341, n_layers, hidden, n_sites], then per token, per layer,
per site (0 = attention input, 1 = FFN/MoE input): float32[hidden].
A partially written file is truncated to whole tokens.

Usage: delta_forest_probe.py ACTS_BIN [REPORT_MD]
"""
import os
import sys
import numpy as np

MAGIC = 0x31544341
BLOCK = 64            # elements per weight-column block
WINDOWS = (8, 16, 32)
CALIB_TOKENS = 200    # fixed per-layer-site INT8 scale from these tokens
T_START = 32          # first token scored (needs 32 predecessors)
FAR = 500             # control: compare against token t-500
SEED = 0              # sign-flip diagonal for the WHT variant
SITE_NAMES = {0: "attn-in", 1: "ffn-in"}


def load(path):
    hdr = np.fromfile(path, dtype="<u4", count=4)
    if len(hdr) < 4 or hdr[0] != MAGIC:
        sys.exit(f"bad header in {path}: {hdr}")
    L, H, S = int(hdr[1]), int(hdr[2]), int(hdr[3])
    per_tok = L * S * H * 4
    T = (os.path.getsize(path) - 16) // per_tok
    mm = np.memmap(path, dtype="<f4", mode="r", offset=16, shape=(T, L, S, H))
    return mm, T, L, S, H


def wht(x):
    """Fast Walsh-Hadamard transform along axis 1 (natural order, unnormalised)."""
    T, H = x.shape
    h = 1
    while h < H:
        x = x.reshape(T, H // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = np.stack([a + b, a - b], axis=2).reshape(T, H)
        h *= 2
    return x


def quantise(x):
    scale = np.abs(x[:CALIB_TOKENS]).max() / 127.0
    return np.clip(np.rint(x / scale), -127, 127).astype(np.int8)


def delta_stats(codes):
    """codes (T,H) int8 -> per-token fractions.  K-window stats cover t >= T_START,
    the far control covers t >= FAR."""
    T, H = codes.shape
    nb = H // BLOCK
    n = T - T_START
    cur = codes[T_START:]
    blk = np.empty((32, n), np.int32)   # changed blocks, offset d = 1..32
    cod = np.empty((32, n), np.int32)   # changed individual codes
    for d in range(1, 33):
        diff = cur != codes[T_START - d:T - d]
        cod[d - 1] = diff.sum(1)
        blk[d - 1] = diff.reshape(n, nb, BLOCK).any(2).sum(1)
    out = {}
    idx = np.arange(n)
    for K in WINDOWS:
        best = (blk[:K] * (H + 1) + cod[:K]).argmin(0)   # fewest blocks, then codes
        out[f"K{K}_blk"] = blk[best, idx] / nb
        out[f"K{K}_cod"] = cod[best, idx] / H
    out["prev_blk"] = blk[0] / nb
    out["prev_cod"] = cod[0] / H
    if T > FAR + 1:
        diff = codes[FAR:] != codes[:T - FAR]
        out["far_blk"] = diff.reshape(T - FAR, nb, BLOCK).any(2).sum(1) / nb
        out["far_cod"] = diff.sum(1) / H
    out["zero_frac"] = np.array([(codes[T_START:] == 0).mean()])
    return out


def pct(a):
    a = np.concatenate([np.ravel(v) for v in a])
    return np.median(a), np.percentile(a, 10), np.percentile(a, 90)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    report = sys.argv[2] if len(sys.argv) > 2 else None
    mm, T, L, S, H = load(path)
    if T <= T_START + 1:
        sys.exit(f"only {T} tokens in dump; need > {T_START + 1}")
    sign = np.random.default_rng(SEED).choice([-1.0, 1.0], size=H).astype(np.float32)
    nonfinite = 0
    res = {}   # (site, mode) -> layer -> stats dict
    for l in range(L):
        for s in range(S):
            x = np.array(mm[:, l, s, :], dtype=np.float32)
            bad = ~np.isfinite(x)
            if bad.any():
                nonfinite += int(bad.sum())
                x[bad] = 0.0
            for mode in ("raw", "wht"):
                z = x if mode == "raw" else wht(x * sign) / np.sqrt(H)
                res.setdefault((s, mode), {})[l] = delta_stats(quantise(z))
        print(f"layer {l + 1}/{L} done", file=sys.stderr, end="\r")
    print(file=sys.stderr)

    lines = []
    P = lines.append
    P(f"# Delta-Forest probe: {os.path.basename(path)}")
    P("")
    P(f"tokens={T} layers={L} hidden={H} sites={S} block={BLOCK} "
      f"calib_tokens={CALIB_TOKENS} scored t>={T_START} far_control=t-{FAR} "
      f"nonfinite_values_zeroed={nonfinite}")
    P("")
    P("Values are fractions of changed 64-element blocks (of 64 per vector) between token t "
      "and the best of its previous K tokens (fewest changed blocks; ties by fewest changed codes), "
      "pooled over all layers and scored tokens.  `codes` = fraction of the 4096 individual INT8 codes "
      "that differ for that same candidate (median).  WHT = signed-diagonal Walsh-Hadamard rotation "
      "before quantisation (natural coefficient order).")
    for s in range(S):
        P("")
        P(f"## site {s} ({SITE_NAMES.get(s, '?')})")
        P("")
        P("| mode | window | blocks median | blocks p10 | blocks p90 | codes median |")
        P("|---|---|---|---|---|---|")
        for mode in ("raw", "wht"):
            st = res[(s, mode)]
            for K in WINDOWS:
                m, lo, hi = pct([st[l][f"K{K}_blk"] for l in range(L)])
                cm = pct([st[l][f"K{K}_cod"] for l in range(L)])[0]
                P(f"| {mode} | best of {K} | {m:.3f} | {lo:.3f} | {hi:.3f} | {cm:.3f} |")
            m, lo, hi = pct([st[l]["prev_blk"] for l in range(L)])
            cm = pct([st[l]["prev_cod"] for l in range(L)])[0]
            P(f"| {mode} | t-1 only | {m:.3f} | {lo:.3f} | {hi:.3f} | {cm:.3f} |")
            if "far_blk" in st[0]:
                m, lo, hi = pct([st[l]["far_blk"] for l in range(L)])
                cm = pct([st[l]["far_cod"] for l in range(L)])[0]
                P(f"| {mode} | control t-{FAR} | {m:.3f} | {lo:.3f} | {hi:.3f} | {cm:.3f} |")
            z = np.mean([st[l]["zero_frac"][0] for l in range(L)])
            P(f"| {mode} | (zero codes, mean over layers) | {z:.3f} | | | |")
        # per-layer ranking by the better of raw/WHT at K=32
        rows = []
        for l in range(L):
            r = [np.median(res[(s, "raw")][l][f"K{K}_blk"]) for K in WINDOWS]
            w = [np.median(res[(s, "wht")][l][f"K{K}_blk"]) for K in WINDOWS]
            rows.append((min(r[-1], w[-1]), l, r, w))
        rows.sort()
        P("")
        P("Per-layer medians of the changed-block fraction (ranked by min(raw, WHT) at K=32):")
        P("")
        P("| rank | layer | raw K8 | raw K16 | raw K32 | wht K8 | wht K16 | wht K32 |")
        P("|---|---|---|---|---|---|---|---|")
        for tag, sel in (("best", rows[:5]), ("worst", rows[-5:])):
            for _, l, r, w in sel:
                P(f"| {tag} | {l} | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} | "
                  f"{w[0]:.3f} | {w[1]:.3f} | {w[2]:.3f} |")
        best = rows[0]
        P("")
        P(f"Best layer at K=32: layer {best[1]} with {best[0]:.3f} of blocks changed; "
          f"layers with median <= 0.30 at K=32 (either mode): "
          f"{sum(1 for r in rows if r[0] <= 0.30)} of {L}.")
    text = "\n".join(lines) + "\n"
    print(text)
    if report:
        with open(report, "w") as f:
            f.write(text)
        print(f"wrote {report}", file=sys.stderr)


if __name__ == "__main__":
    main()
