#!/usr/bin/env python3
"""P5: choose the DFlash draft count k and the min-p cut from one --dflash 7 --dflash-log run (ie-mimo26-run).

  dflash_policy.py <run.log> <draftlog.txt> [--cost STEP1,CROW,DRAFTER,CTX]

draftlog: one line per DFlash pass, "accepted n_drafted p1 .. p7" (the drafter's top-1 softmax probability per row).
Acceptance is a prefix and the verify is causal, so submitting only the first m drafts accepts min(accepted, m) -- the
same tokens (up to multi-row rounding at near-ties). The pass cost model is fitted on the run itself: per prompt,
decode time = passes * (drafter + context + step1) + (rows - passes) * c_row, least squares over the prompts, with the
drafter and context times the run measured. Prints the fit and ms/token for every (k, min-p).
"""
import re
import sys

import numpy as np

log, dlog = sys.argv[1], sys.argv[2]
t = open(log, errors="replace").read()
dec = [float(x) for x in re.findall(r"^decode: ([\d.]+) ms/token", t, re.M)]
gen = [int(x) for x in re.findall(r"^generated: (\d+) tokens", t, re.M)]
dfl = re.findall(r"^dflash: (\d+) passes \((\d+) rows, (\d+) drafts accepted = [\d.]+ per pass\), drafter ([\d.]+) ms/pass, context ([\d.]+) ms/pass", t, re.M)
assert len(dec) == len(gen) == len(dfl), (len(dec), len(gen), len(dfl))
# fit step1 and c_row: total_ms - passes * (drafter + context) = passes * step1 + (rows - passes) * c_row
A, b = [], []
for d, g, (np_, nr, na, dm, cm) in zip(dec, gen, dfl):
    np_, nr, dm, cm = int(np_), int(nr), float(dm), float(cm)
    if np_ == 0: continue
    A.append([np_, nr - np_]); b.append(d * g - np_ * (dm + cm))
(step1, c_row), *_ = np.linalg.lstsq(np.array(A, float), np.array(b, float), rcond=None)
drafter = float(np.mean([float(x[3]) for x in dfl])); ctx = float(np.mean([float(x[4]) for x in dfl]))
# a fixed-k run cannot separate step1 from c_row (every pass has k + 1 rows): --cost STEP1,CROW,DRAFTER,CTX overrides
if "--cost" in sys.argv:
    step1, c_row, drafter, ctx = (float(x) for x in sys.argv[sys.argv.index("--cost") + 1].split(","))
print(f"{len(dec)} prompts, {sum(int(x[0]) for x in dfl)} passes; fit: one-row step {step1:.1f} ms, +{c_row:.1f} ms per extra verify row; "
      f"drafter {drafter:.1f} + context {ctx:.1f} ms per pass; measured {sum(d * g for d, g in zip(dec, gen)) / sum(gen):.1f} ms/token")

passes = []
for line in open(dlog):
    f = line.split()
    acc, nd, ps = int(f[0]), int(f[1]), [float(x) for x in f[2:]]
    passes.append((acc, nd, ps))


def simulate(k, tau):
    ms = tok = 0.0
    for acc, nd, ps in passes:
        m = 0
        while m < min(k, nd) and ps[m] >= tau:
            m += 1
        a = min(acc, m)
        ms += drafter + ctx + step1 + m * c_row
        tok += a + 1
    return ms / tok


plain = step1   # a plain one-row step, no drafter
print(f"plain (model): {plain:.1f} ms/token")
taus = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
print("k \\ min-p " + " ".join(f"{x:6.1f}" for x in taus))
best = (1e9, None)
for k in range(1, 8):
    row = [simulate(k, tau) for tau in taus]
    for tau, v in zip(taus, row):
        if v < best[0]: best = (v, (k, tau))
    print(f"k={k}       " + " ".join(f"{v:6.1f}" for v in row))
print(f"best: k={best[1][0]} min-p {best[1][1]} -> {best[0]:.1f} ms/token (model), x{plain / best[0]:.2f} vs plain")
# calibration: acceptance of a reached draft row vs its probability
edges = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
hit = np.zeros(len(edges) - 1); cnt = np.zeros(len(edges) - 1)
for acc, nd, ps in passes:
    for i in range(min(nd, len(ps))):
        if i > acc: break   # rows after the first rejection were never judged on their own
        j = int(np.searchsorted(edges, ps[i], side="right")) - 1
        cnt[j] += 1; hit[j] += i < acc
print("calibration (reached rows): " + ", ".join(f"p[{edges[j]:.1f},{min(edges[j + 1], 1):.1f}) {int(cnt[j])} rows {100 * hit[j] / max(cnt[j], 1):.0f}% accepted" for j in range(len(cnt))))
