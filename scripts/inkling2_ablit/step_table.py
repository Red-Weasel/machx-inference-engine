#!/usr/bin/env python3
"""Counts-only table for refusal / steering sets: harmful prompts (answered / declined / still reasoning / loops) and
harmless + coding prompts (declined, unfinished, loops, reasoning and answer tokens, KL, code-block agreement).

  python step_table.py --harmful H1.jsonl ... --benign B1.jsonl ... [--base none]

Generation files from ie-inkling2-run --gen-out (several candidates may be mixed; grouped by "candidate").
Benign agreement is measured against --base. COUNTS ONLY: no reply text is printed.
"""
import argparse, json, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import resolve, is_loop  # noqa: E402

END = 200010


def load(files):
    by = {}
    for f in files:
        for l in open(f, encoding="utf-8", errors="replace"):
            if l.strip(): r = json.loads(l); by.setdefault(r["candidate"], {})[r["name"]] = r
    return by


def tl(ids): return ids.index(END) if END in ids else len(ids)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--harmful", nargs="*", default=[]); ap.add_argument("--benign", nargs="*", default=[])
    ap.add_argument("--base", default="none"); a = ap.parse_args()
    H = load(a.harmful)
    if H: print(f"{'harmful':10s} {'n':>3s} {'answered':>8s} {'declined':>8s} {'reasoning':>9s} {'loops':>5s} {'tokens':>6s}")
    for c, rs in H.items():
        k = {"complied": 0, "declined": 0, "unfinished": 0}
        for r in rs.values():
            if not r["text"].strip() and not r.get("tool_calls"): k["unfinished"] += 1
            else: k[resolve(r["text"])[0]] += 1
        print(f"{c:10s} {len(rs):3d} {k['complied']:8d} {k['declined']:8d} {k['unfinished']:9d} {sum(is_loop(r['ids']) for r in rs.values()):5d} "
              f"{st.mean(len(r['ids']) for r in rs.values()):6.0f}")
    B = load(a.benign); base = B.get(a.base, {})
    if B: print(f"{'benign':10s} {'n':>3s} {'decl':>4s} {'unfin':>5s} {'loops':>5s} {'reason':>6s} {'answer':>6s} {'ans/base':>8s} {'code=':>5s} {'KL16':>6s} {'KLmax':>6s}")
    for c, rs in B.items():
        dec = sum(1 for r in rs.values() if r["text"].strip() and resolve(r["text"])[0] == "declined")
        un = sum(1 for r in rs.values() if not r["text"].strip() and not r.get("tool_calls"))
        rat = [len(r["text"]) / len(base[n]["text"]) for n, r in rs.items() if n in base and base[n]["text"]]
        same = sum(("```" in r["text"]) == ("```" in base[n]["text"]) for n, r in rs.items() if n in base)
        kl = [sum(r["kl"]) / len(r["kl"]) for r in rs.values() if r.get("kl")]
        print(f"{c:10s} {len(rs):3d} {dec:4d} {un:5d} {sum(is_loop(r['ids']) for r in rs.values()):5d} {st.mean(tl(r['ids']) for r in rs.values()):6.0f} "
              f"{st.mean(len(r['ids']) - tl(r['ids']) for r in rs.values()):6.0f} {st.median(rat) if rat else float('nan'):8.2f} {same:5d} "
              f"{st.mean(kl) if kl else float('nan'):6.3f} {max(kl) if kl else float('nan'):6.3f}")


if __name__ == "__main__": main()
