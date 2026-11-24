"""Counts only: records, rendered length (Inkling tokens) and what the windowing rule does, per data file.

  xpu_env.sh venv/bin/python -I data_stats.py MODEL_DIR MAX_LEN FILE [FILE ...]
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from inkling2.data import load_file, load_sample, newest_sample

model, max_len, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
tok = AutoTokenizer.from_pretrained(model)
sp = newest_sample()
tools = load_sample(sp)["tools"]
print(f"tools from {os.path.basename(sp)}: {len(tools)}")
for f in files:
    t = time.time()
    ex, st = load_file(f, tok, tools, max_len)
    L = sorted(len(e["ids"]) for e in ex)
    st["example_median_tokens"] = L[len(L) // 2]
    st["render_seconds"] = round(time.time() - t)
    print(json.dumps(st), flush=True)
