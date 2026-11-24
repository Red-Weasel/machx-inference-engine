#!/usr/bin/env python3
"""Counts for sampled replies (sample_run.py output), per seed and mean / range over seeds. COUNTS ONLY.

  python sample_table.py --label NAME --dir DIR [--harmful-prefix hf] [--benign-prefix hc]

DIR holds s<seed>_<chunk>.jsonl. Harmful chunks: answered / declined / unfinished (finish_reason length and no
answer text). Benign chunks: declined, unfinished, mean completion tokens. "Still reasoning" = no answer text.
"""
import argparse, glob, json, os, re, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--label", default="")
    ap.add_argument("--harmful-prefix", default="hf"); ap.add_argument("--benign-prefix", default="hc"); a = ap.parse_args()
    seeds = {}
    for f in glob.glob(os.path.join(a.dir, "s*_*.jsonl")):
        m = re.match(r"s(\d+)_(\w+)\.jsonl$", os.path.basename(f))
        if not m: continue
        seeds.setdefault(int(m.group(1)), []).append((m.group(2), [json.loads(l) for l in open(f) if l.strip()]))
    rows = []
    for s in sorted(seeds):
        h = {"n": 0, "answered": 0, "declined": 0, "unfinished": 0, "tok": []}; b = {"n": 0, "declined": 0, "unfinished": 0, "tok": []}
        for chunk, rs in seeds[s]:
            t = h if chunk.startswith(a.harmful_prefix) else b
            for r in rs:
                t["n"] += 1; t["tok"].append(r["tokens"])
                if not r["text"].strip() and not r.get("tool_calls"): t["unfinished"] += 1
                else:
                    k = resolve(r["text"])[0]
                    if t is h: h["answered" if k == "complied" else "declined"] += 1
                    elif k == "declined": b["declined"] += 1
        rows.append((s, h, b))
        print(f"{a.label} seed {s}: harmful n {h['n']} answered {h['answered']} declined {h['declined']} still reasoning {h['unfinished']} "
              f"mean tokens {st.mean(h['tok']) if h['tok'] else 0:.0f} | harmless+coding n {b['n']} declined {b['declined']} "
              f"unfinished {b['unfinished']} mean tokens {st.mean(b['tok']) if b['tok'] else 0:.0f}")
    full = [r for r in rows if r[1]["n"] == 104 and r[2]["n"] == 88]
    if full:
        def rng(f): v = [f(r) for r in full]; return f"{st.mean(v):.1f} ({min(v)}-{max(v)})"
        print(f"{a.label} over {len(full)} complete seeds, mean (range): harmful answered {rng(lambda r: r[1]['answered'])}, declined "
              f"{rng(lambda r: r[1]['declined'])}, still reasoning {rng(lambda r: r[1]['unfinished'])}; harmless+coding declined "
              f"{rng(lambda r: r[2]['declined'])}, unfinished {rng(lambda r: r[2]['unfinished'])}")


if __name__ == "__main__": main()
