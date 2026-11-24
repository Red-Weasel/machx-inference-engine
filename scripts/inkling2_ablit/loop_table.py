#!/usr/bin/env python3
"""One table of every candidate in the given generation files against the baseline (candidate "none").

  python loop_table.py --answers answers.json G.jsonl [G2 ...]

Per candidate: n, unfinished (no answer text), loops, correct, gained / lost correct vs the baseline on the same
prompts, mean reasoning tokens, mean total tokens, mean KL over the 16 reference positions (when recorded), and on the
prompts without an answer key (harmless / coding) how many were declined (score.resolve).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_data import correct, think_len  # noqa: E402
from score import is_loop, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--answers", required=True); ap.add_argument("gens", nargs="+")
    a = ap.parse_args()
    A = {p["name"]: p["answer"] for p in json.load(open(a.answers))}
    by = {}
    for f in a.gens:
        for l in open(f, encoding="utf-8", errors="replace"):
            if l.strip(): r = json.loads(l); by.setdefault(r["candidate"], {})[r["name"]] = r
    base = by.get("none", {})
    print(f"{'candidate':10s} {'n':>3s} {'unfin':>5s} {'loops':>5s} {'correct':>7s} {'+':>3s} {'-':>3s} {'reason':>6s} {'total':>6s} {'KL16':>6s} {'nokey':>5s} {'decl':>4s}")
    for c, rs in by.items():
        ok = {n: correct(r["text"], A[n]) for n, r in rs.items() if n in A}
        g = sum(1 for n in ok if ok[n] and n in base and not correct(base[n]["text"], A[n]))
        lo = sum(1 for n in ok if not ok[n] and n in base and correct(base[n]["text"], A[n]))
        kl = [sum(r["kl"]) / len(r["kl"]) for r in rs.values() if r.get("kl")]
        n = len(rs)
        nk = [r for nm, r in rs.items() if nm not in A]               # prompts without an answer key (harmless / coding)
        dec = sum(1 for r in nk if r["text"].strip() and resolve(r["text"])[0] == "declined")
        print(f"{c:10s} {n:3d} {sum(not r['text'].strip() for r in rs.values()):5d} {sum(is_loop(r['ids']) for r in rs.values()):5d} "
              f"{sum(ok.values()):7d} {g:3d} {lo:3d} {sum(think_len(r['ids']) for r in rs.values()) / n:6.0f} "
              f"{sum(len(r['ids']) for r in rs.values()) / n:6.0f} {(sum(kl) / len(kl)) if kl else float('nan'):6.3f} {len(nk):5d} {dec:4d}")


if __name__ == "__main__": main()
