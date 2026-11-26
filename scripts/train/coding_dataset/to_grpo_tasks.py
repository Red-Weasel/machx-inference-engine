#!/usr/bin/env python3
"""The checked coding tasks (~/models/inkling2-train/data/checked) as GRPO task files for machx-train.

  to_grpo_tasks.py [--checked DIR] --out-train train.grpo.jsonl --out-heldout heldout.grpo.jsonl

Each line: {"id", "family", "language", "messages": [user], "reward": {"type": "tests", ...}}. The user turn is the
task statement, plus (for code) one line saying how to hand the file back, since a single reply has no file tools.
The tests and the reference answer stay out of "messages"; the reference answer is not written at all (GRPO trains
only on the model's own replies).
"""
import argparse
import json
import os
import shlex

ap = argparse.ArgumentParser()
ap.add_argument("--checked", default=os.path.expanduser("~/models/inkling2-train/data/checked"))
ap.add_argument("--out-train", required=True)
ap.add_argument("--out-heldout", required=True)
a = ap.parse_args()

FENCE = {"python": "python", "javascript": "javascript", "cpp": "cpp", "rust": "rust"}
counts = {"train": 0, "heldout": 0}
outs = {"train": open(a.out_train, "w"), "heldout": open(a.out_heldout, "w")}
for line in open(os.path.join(a.checked, "manifest.jsonl")):
    m = json.loads(line)
    d = os.path.join(a.checked, "tasks", m["id"])
    statement = open(os.path.join(d, "statement.md")).read().rstrip()
    files = {}
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if name in ("statement.md", m["answer_file"], "mutants") or not os.path.isfile(p):
            continue
        files[name] = open(p).read()
    check = [shlex.split(c) for c in m["verification"].split(" && ")]
    if m["language"] == "text":
        user, extract = statement, "answer_line"
    else:
        user = (statement + f"\n\nReply with the complete contents of `{m['answer_file']}` in one fenced code block "
                f"(```{FENCE[m['language']]}).")
        extract = "fence"
    rec = {"id": m["id"], "family": m["family"], "language": m["language"],
           "messages": [{"role": "user", "content": user}],
           "reward": {"type": "tests", "language": m["language"], "answer_file": m["answer_file"], "files": files,
                      "check": check, "timeout": 60, "extract": extract}}
    outs[m["split"]].write(json.dumps(rec) + "\n")
    counts[m["split"]] += 1
print(json.dumps(counts))
