#!/usr/bin/env python3
"""Dream's first request for an ordinary conversation, taken inside the same sandbox as the runs: with and without the page."""
import json, os, shutil, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for name, text in (("nopage", None), ("page", open(C.PAGE).read().strip())):
    slot = 40
    cmd, w, rec, home, tmp = C.dream_cmd(slot, 1, f"dream.request_sample --out {os.path.join(C.SLOTS, f'r{slot:02d}', 's.json')} --workspace {os.path.join(C.SLOTS, f'w{slot:02d}')} --root {os.path.join(C.SLOTS, f'r{slot:02d}', 'root')} --prompt Hello.")
    for d in (w, rec, home, tmp):
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d, mode=0o700)
    os.makedirs(C.LEASES, mode=0o700, exist_ok=True)
    os.makedirs(os.path.join(rec, "root", "memory"))
    if text:
        open(os.path.join(rec, "root", "memory", "INSTRUCTIONS.md"), "w").write(text + "\n")
    print(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())
    shutil.copy(os.path.join(rec, "s.json"), os.path.join(out, f"sample_{name}.json"))
    for d in (w, rec, home, tmp):
        shutil.rmtree(d, ignore_errors=True)
