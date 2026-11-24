#!/usr/bin/env python3
"""Inkling 2 Mini abliteration: the held-out numbers for the report / README. COUNTS ONLY.

  python final_numbers.py --harmful-orig F --harmful-abl F --harmless-code F --abl-name NAME [--json-out F]

Harmful: complied / declined / unfinished (score.py rules; unclear answers resolved by their first sentence), on all
104 test prompts and split into the 48 used to pick candidates (harmful_test_0..47) and the 56 never used (48..103).
Harmless (harmless_test_*) and coding (code_*): declined, loops, finished turns, mean reply tokens, KL (first content
position / mean of 16) of the ablated model against the original along the original's own reply.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import resolve, is_loop  # noqa: E402


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8", errors="replace") if l.strip()]


def harm(rs):
    t = {"complied": 0, "declined": 0, "unfinished": 0, "by_opening": 0, "n": len(rs), "loops": 0, "mean_tokens": 0.0}
    for r in rs:
        t["loops"] += is_loop(r["ids"])
        if not r["text"].strip() and not r.get("tool_calls"): t["unfinished"] += 1; continue
        k, op = resolve(r["text"]); t[k] += 1; t["by_opening"] += op
    t["mean_tokens"] = float(np.mean([len(r["ids"]) for r in rs])) if rs else 0.0
    return t


def benign(rs, ref_ids):
    t = {"n": len(rs), "declined": sum(1 for r in rs if r["text"].strip() and resolve(r["text"])[0] == "declined"),
         "unfinished": sum(1 for r in rs if not r["text"].strip() and not r.get("tool_calls")),
         "loops": sum(is_loop(r["ids"]) for r in rs), "ended": sum(r["ended"] for r in rs),
         "mean_tokens": float(np.mean([len(r["ids"]) for r in rs])) if rs else 0.0}
    kls = [(r["kl"], ref_ids[r["name"]]) for r in rs if r.get("kl")]
    if kls:
        def first(kl, ids):
            for k, i in zip(kl, ids):
                if i < 199998: return k
            return kl[-1]
        f = [first(k, i) for k, i in kls]; m = [float(np.mean(k)) for k, _ in kls]
        t.update(kl_first_mean=float(np.mean(f)), kl_first_max=float(np.max(f)), kl_mean=float(np.mean(m)), kl_mean_max=float(np.max(m)))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful-orig", required=True); ap.add_argument("--harmful-abl", required=True)
    ap.add_argument("--harmless-code", required=True); ap.add_argument("--abl-name", required=True); ap.add_argument("--json-out")
    a = ap.parse_args()
    out = {}
    idx = lambda r: int(r["name"].rsplit("_", 1)[1])
    for lab, f, cand in (("original", a.harmful_orig, None), ("ablated", a.harmful_abl, a.abl_name)):
        rs = [r for r in rows(f) if cand is None or r["candidate"] == cand]
        out[f"harmful_{lab}"] = {"all": harm(rs), "selection_0_47": harm([r for r in rs if idx(r) < 48]),
                                 "untouched_48_103": harm([r for r in rs if idx(r) >= 48])}
    hc = rows(a.harmless_code)
    orig = [r for r in hc if r["candidate"] == "none"]; abl = [r for r in hc if r["candidate"] == a.abl_name]
    ref = {r["name"]: r["ids"] for r in orig}
    for lab, rs in (("original", orig), ("ablated", abl)):
        out[f"harmless_{lab}"] = benign([r for r in rs if r["name"].startswith("harmless")], ref)
        out[f"code_{lab}"] = benign([r for r in rs if r["name"].startswith("code")], ref)
    # per coding prompt: same kind of answer? (both have a code block / neither) and the length ratio
    o = {r["name"]: r for r in orig if r["name"].startswith("code")}; b = {r["name"]: r for r in abl if r["name"].startswith("code")}
    same_kind = sum(("```" in o[n]["text"]) == ("```" in b[n]["text"]) for n in o if n in b)
    ratios = [len(b[n]["text"]) / max(1, len(o[n]["text"])) for n in o if n in b and o[n]["text"]]
    out["code_compare"] = {"n": len(o), "same_code_block_presence": same_kind,
                           "answer_length_ratio_median": float(np.median(ratios)) if ratios else None,
                           "answer_length_ratio_min": float(np.min(ratios)) if ratios else None,
                           "answer_length_ratio_max": float(np.max(ratios)) if ratios else None}
    print(json.dumps(out, indent=1))
    if a.json_out: json.dump(out, open(a.json_out, "w"), indent=1)


if __name__ == "__main__": main()
