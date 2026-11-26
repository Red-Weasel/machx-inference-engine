#!/usr/bin/env python3
"""Compare record folders: table.py LABEL [LABEL ...]. Passed / failed / timed out, tool errors, tokens; paired wins."""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def load(label):
    out = {}
    base = os.path.join(C.RECORDS, label)
    for k in sorted(os.listdir(base)):
        for n in sorted(os.listdir(os.path.join(base, k))):
            d = os.path.join(base, k, n)
            if os.path.exists(os.path.join(d, "grade.json")):
                g = json.load(open(os.path.join(d, "grade.json")))
                try:
                    r = json.load(open(os.path.join(d, "result.json")))
                except (OSError, ValueError):
                    r = {}
                out[(k, n)] = (g, r)
    return out


runs = {x: load(x) for x in sys.argv[1:]}
print("label runs passed failed_tests no_answer timed_out other_end tool_calls failed_calls nonzero_exits out_tokens/run")
for x, rs in runs.items():
    c = collections.Counter()
    for g, r in rs.values():
        c["passed"] += g["passed"]
        if not g["passed"]:
            e = g.get("ended")
            key = "timed_out" if e == "time_limit" or g["reason"] == "test_timeout" else "failed_tests" if g["reason"] == "test_failed" \
                else "no_answer" if g["reason"] in ("no_answer_file", "no_final_answer", "answer_too_large") else "other_end"
            c[key] += 1
        for f, t in (("tool_calls", "tool_calls"), ("failed_tool_calls", "failed_calls"), ("nonzero_exits", "nonzero"), ("output_tokens", "out")):
            c[t] += r.get(f) or 0
    n = max(1, len(rs))
    print(x, len(rs), c["passed"], c["failed_tests"], c["no_answer"], c["timed_out"], c["other_end"], c["tool_calls"], c["failed_calls"], c["nonzero"], c["out"] // n)
    fam = collections.defaultdict(lambda: [0, 0])
    for (k, _), (g, _) in rs.items():
        f = k.rsplit("_", 1)[0]
        fam[f][0] += g["passed"]
        fam[f][1] += 1
    print("   ", " ".join(f"{f}={p}/{t}" for f, (p, t) in sorted(fam.items())))
if len(runs) == 2:
    (a, ra), (b, rb) = runs.items()
    both = set(ra) & set(rb)
    only_a = sum(ra[k][0]["passed"] and not rb[k][0]["passed"] for k in both)
    only_b = sum(rb[k][0]["passed"] and not ra[k][0]["passed"] for k in both)
    print(f"paired on {len(both)}: only {a} passed {only_a}; only {b} passed {only_b}")
