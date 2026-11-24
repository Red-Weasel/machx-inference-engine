#!/usr/bin/env python3
"""Inkling 2 Mini "conclude vs keep re-checking": classify replies and build recording inputs (neutral prompts only).

  python loop_data.py score  --gens G.jsonl [G2 ...] --answers answers.json [--label L]
  python loop_data.py inputs --gens G.jsonl [...] --prompts P.jsonl --out DIR [--span 32]

score: per reply: closed (the thinking block ended and an answer followed), unfinished (no answer text at the cap),
loop (score.is_loop), correct (an expected answer string appears in the answer text; numbers also without commas).
Prints counts, mean reasoning tokens (up to the first <|end_message|>) and mean total tokens.

inputs: three .jsonl batch files for ie-inkling2-run --batch --acts-last SPAN --acts-mean (each line = the prompt plus
append_ids = a prefix of the model's own reply, so the recorder's last SPAN positions are a stretch of the reasoning):
  close.jsonl   closed replies: the reply up to the end of its thinking (the SPAN tokens just before it closes)
  open.jsonl    unfinished replies: the reply up to the same token count as a matched closed reply's close point
                (matched by rank of length; the stretch at a comparable depth that did NOT close)
  loopin.jsonl  looping replies: the reply up to its end (the SPAN tokens inside the loop)
  mid.jsonl     closed replies cut at half their reasoning depth (within-reply control)
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import is_loop  # noqa: E402

END, THINK = 200010, 200008


def think_len(ids):
    return ids.index(END) if END in ids else len(ids)


def load(files):
    rs = []
    for f in files: rs += [json.loads(l) for l in open(f, encoding="utf-8", errors="replace") if l.strip()]
    return rs


def correct(text, answers):
    t = text.lower().replace(",", "")
    return any(str(a).lower().replace(",", "") in t for a in answers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "inputs"]); ap.add_argument("--gens", nargs="+", required=True)
    ap.add_argument("--answers"); ap.add_argument("--prompts", nargs="+"); ap.add_argument("--out"); ap.add_argument("--span", type=int, default=32)
    ap.add_argument("--label", default="")
    a = ap.parse_intermixed_args()
    rs = load(a.gens)
    if a.cmd == "score":
        ans = {p["name"]: p["answer"] for p in json.load(open(a.answers))}
        by = {}
        for r in rs: by.setdefault(r["candidate"], []).append(r)
        for c, rows in by.items():
            closed = [r for r in rows if r["text"].strip()]
            n_ok = sum(correct(r["text"], ans[r["name"]]) for r in rows if r["name"] in ans)
            print(f"{a.label} {c}: n {len(rows)}, closed {len(closed)}, unfinished {len(rows) - len(closed)}, loops {sum(is_loop(r['ids']) for r in rows)}, "
                  f"correct {n_ok} of {sum(r['name'] in ans for r in rows)}, mean reasoning tokens {sum(think_len(r['ids']) for r in rows) / len(rows):.0f}, "
                  f"mean total tokens {sum(len(r['ids']) for r in rows) / len(rows):.0f}")
        return
    P = {}
    for f in a.prompts: P.update({json.loads(l)["name"]: json.loads(l)["text"] for l in open(f) if l.strip()})
    os.makedirs(a.out, exist_ok=True)
    closed = [r for r in rs if r["text"].strip() and END in r["ids"] and think_len(r["ids"]) >= a.span + 4]
    unfin = [r for r in rs if not r["text"].strip()]
    loops = [r for r in rs if is_loop(r["ids"])]
    cl_sorted = sorted(think_len(r["ids"]) for r in closed)
    def w(name, items):
        with open(os.path.join(a.out, name), "w") as f:
            for r, n in items: f.write(json.dumps({"name": r["name"], "text": P[r["name"]], "append_ids": r["ids"][:n]}) + "\n")
        print(name, len(items))
    w("close.jsonl", [(r, think_len(r["ids"])) for r in closed])
    # matched depth: unfinished reply i gets the close point of the closed reply at the same rank (longest ones)
    depths = cl_sorted[-len(unfin):] if unfin else []
    w("open.jsonl", [(r, max(a.span + 4, min(d, len(r["ids"]) - 1))) for r, d in zip(sorted(unfin, key=lambda r: r["name"]), depths)])
    w("loopin.jsonl", [(r, len(r["ids"])) for r in loops])
    # within-reply control: the same closed replies at half their reasoning depth (still reasoning, will conclude later)
    w("mid.jsonl", [(r, max(a.span + 4, think_len(r["ids"]) // 2)) for r in closed])


if __name__ == "__main__": main()
