#!/usr/bin/env python3
"""Hy4-flash abliteration search: score an ie-hy4flash-run --ablate-list --gen-out file, one line per candidate.

  python score_sweep.py sweep.jsonl [--show 3] [--only NAME]

A prompt named harmful_* / harmless_* is scored with behavior.py's rules (a refusal phrase anywhere, or an answer that
opens by judging the request). Per candidate: how many harmful prompts are still declined (the fewer the better), how many
harmless ones are declined or degenerate (damage), and the first words of a few harmful answers.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from behavior import classify, is_degenerate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file"); ap.add_argument("--show", type=int, default=0); ap.add_argument("--only")
    a = ap.parse_args()
    order, by = [], {}
    for line in open(a.file, encoding="utf-8", errors="replace"):      # (a generation cut inside a multi-byte character)
        if not line.strip(): continue
        r = json.loads(line); c = r.get("candidate", "")
        if c not in by: by[c] = []; order.append(c)
        by[c].append(r)
    print(f"{'candidate':30s} {'harmful: complied / unclear / declined':>40s} {'harmless declined':>18s} {'degenerate':>11s}")
    for c in order:
        if a.only and c != a.only: continue
        h = [r for r in by[c] if r["name"].startswith("harmful")]; b = [r for r in by[c] if r["name"].startswith("harmless")]
        ch = [classify(r["text"]) for r in h]; cb = [classify(r["text"]) for r in b]
        deg = sum(is_degenerate(r["ids"]) for r in by[c])
        print(f"{c:30s} {ch.count('complied'):>22d} / {ch.count('unclear'):>3d} / {ch.count('declined'):>3d} of {len(h):<3d} {cb.count('declined'):>11d} of {len(b):<3d} {deg:>11d}")
        for r, k in list(zip(h, ch))[:a.show]: print(f"      [{k:8s}] {r['name']}: {r['text'][:150]!r}")
        for r, k in list(zip(b, cb))[:min(a.show, 2)]: print(f"      [{k:8s}] {r['name']}: {r['text'][:150]!r}")

if __name__ == "__main__": main()
