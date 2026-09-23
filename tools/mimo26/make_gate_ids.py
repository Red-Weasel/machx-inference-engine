#!/usr/bin/env python3
"""P2 gate clause (b) inputs (docs/mimo26/00_PORT_PLAN.md): the teacher-forced gate sequences.

  make_gate_ids.py <model_dir> <out ids.txt> <run.log>... [--wiki FILE --wiki-tokens N]

Each ie-mimo26-run log contributes one sequence: its prompt ids followed by its generated ids (the eos, if any,
dropped). --wiki adds the first N tokens of FILE tokenized with the checkpoint's own tokenizer.json (HF tokenizers).
"""
import re
import sys

args = sys.argv[1:]
wiki, wiki_n = None, 512
if "--wiki" in args:
    i = args.index("--wiki"); wiki = args[i + 1]; del args[i:i + 2]
if "--wiki-tokens" in args:
    i = args.index("--wiki-tokens"); wiki_n = int(args[i + 1]); del args[i:i + 2]
model, out, logs = args[0], args[1], args[2:]

lines = []
for log in logs:
    txt = open(log, encoding="utf-8", errors="replace").read()
    p = re.search(r"^prompt: \d+ tokens \[([^\]]*)\]", txt, re.M)
    g = re.search(r"^generated: \d+ tokens \[([^\]]*)\]", txt, re.M)
    if not p or not g:
        sys.exit(f"{log}: no prompt/generated lines")
    ids = [int(x) for x in p.group(1).split(",") if x.strip()] + [int(x) for x in g.group(1).split(",") if x.strip()]
    while ids and ids[-1] in (151643, 151645, 151672):
        ids.pop()
    lines.append(ids)
if wiki:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{model}/tokenizer.json")
    ids = tok.encode(open(wiki, encoding="utf-8").read()[:20000], add_special_tokens=False).ids[:wiki_n]
    lines.append(ids)
with open(out, "w") as f:
    for ids in lines:
        f.write(" ".join(str(i) for i in ids) + "\n")
print(f"wrote {len(lines)} sequences ({', '.join(str(len(l)) for l in lines)} tokens) to {out}")
