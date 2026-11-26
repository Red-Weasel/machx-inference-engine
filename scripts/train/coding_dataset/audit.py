#!/usr/bin/env python3
"""Audit a Dream test run from what the model actually met in each session: every setup problem (not a model mistake).

  audit.py LABEL        (records in dream-coding-work/records/LABEL)

Per session it flags: no_file (the model looked for the task's file and it was not there), trimmed (Dream cut old
messages to fit the context: "[elided:"), tmp_blocked ("Permission denied" under /tmp), cut_at_limit (a reply stopped
at the per-request token limit), lease (Dream refused a request), error (the session ended in an error), and
prompt_near_ctx (its largest prompt was within 10 % of the engine's slot context). Prints counts, pass rates of the
flagged sessions, and one example of each flag.
"""
import collections
import glob
import json
import os
import re
import sys

R = os.path.join(os.path.join(os.environ.get("IE_DATA_DRIVE", os.path.join("/media", os.environ.get("USER", ""), "Secondary Drive")), "dream-coding-work"), "records")
L = os.path.join(os.path.join(os.environ.get("IE_DATA_DRIVE", os.path.join("/media", os.environ.get("USER", ""), "Secondary Drive")), "dream-coding-work"), "logs")
label = sys.argv[1]
ctx = None
log = os.path.join(L, label + ".ie.log")
if os.path.exists(log):
    m = re.findall(r"serving .*\(ctx (\d+), parallel", open(log, errors="replace").read())
    ctx = int(m[-1]) if m else None
flags, passed, example, n = collections.Counter(), collections.Counter(), {}, 0
for g in sorted(glob.glob(os.path.join(R, label, "*", "*", "grade.json"))):
    d = os.path.dirname(g)
    n += 1
    G = json.load(open(g))
    s = open(os.path.join(d, "session.json")).read() if os.path.exists(os.path.join(d, "session.json")) else ""
    r = json.load(open(os.path.join(d, "result.json"))) if os.path.exists(os.path.join(d, "result.json")) else {}
    f = []
    if re.search(r"No file at [^\"]*(solution|answer)", s):
        f.append("no_file")
    if "[elided:" in s:
        f.append("trimmed")
    if "Permission denied" in s and "/tmp" in s:
        f.append("tmp_blocked")
    if '"finish_reason": "length"' in s or "cut off at the token limit" in s or "reply was cut" in s:
        f.append("cut_at_limit")
    if "lease_refused" in s or "lease_refused" in (r.get("error") or ""):
        f.append("lease")
    if r.get("ended") == "error":
        f.append("error")
    if ctx and (r.get("peak_prompt_tokens") or 0) > 0.9 * ctx:
        f.append("prompt_near_ctx")
    for x in f:
        flags[x] += 1
        passed[x] += bool(G["passed"])
        example.setdefault(x, os.path.relpath(d, R))
print(f"{label}: {n} sessions audited; engine slot context {ctx}")
if os.path.exists(log):
    cut = len(re.findall(r"reply (\d+) tokens, finish length", open(log, errors="replace").read()))
    print(f"  replies cut at the per-request token limit (engine log, whole run): {cut}")
if not flags:
    print("  no setup problem found in any session")
for x, c in flags.most_common():
    print(f"  {x:16s} {c:4d} sessions ({passed[x]} of them passed)   e.g. {example[x]}")
