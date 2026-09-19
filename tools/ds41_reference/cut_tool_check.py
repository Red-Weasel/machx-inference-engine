#!/usr/bin/env python3
"""docs/deepseek41/100: ds41_cut_tool_name through ie-ds41-protocol-fixture.
usage: cut_tool_check.py <path to ie-ds41-protocol-fixture>"""
import json, subprocess, sys

D = "<｜DSML｜"
CASES = [
    ("plain prose, no call", "Here is the plan.", ""),
    ("cut inside a parameter", f"Writing it now.\n\n{D} calls>\n{D} invoke name=\"write_file\">\n{D} parameter name=\"path\" string=\"true\">a.js", "write_file"),
    ("cut before the name is complete", f"Ok.\n\n{D} calls>\n{D} invoke name=\"write_f", "unknown"),
    ("cut right at the calls marker", f"Ok.\n\n{D} calls>\n", "unknown"),
    ("partial marker only", "Ok.\n\n<｜DS", ""),
    ("second call cut", f"{D} calls>\n{D} invoke name=\"read_file\">\n{D} parameter name=\"path\" string=\"true\">x</{D} parameter>\n</{D} invoke>\n{D} invoke name=\"run_bash\">\n", "run_bash"),
]
lines = "\n".join(json.dumps({"partial": t}) for _, t, _ in CASES) + "\n"
out = subprocess.run([sys.argv[1]], input=lines, capture_output=True, text=True, check=True).stdout.splitlines()
fail = 0
for (name, text, want), got in zip(CASES, out):
    r = json.loads(got)
    ok = r["cut_tool"] == want and D not in r["content"]
    fail += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: cut_tool={r['cut_tool']!r} (want {want!r})")
print(f"\nCUT TOOL CHECK: {'FAIL' if fail else 'PASS'} ({len(CASES) - fail}/{len(CASES)})")
sys.exit(1 if fail else 0)
