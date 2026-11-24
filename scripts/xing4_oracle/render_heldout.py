import os, sys, json, numpy as np
sys.path.insert(0, "scripts/xing4_oracle")
from transformers import AutoTokenizer
import gptq
M = os.path.join(os.environ.get("IE_DATA_DRIVE", os.path.expanduser("~/data")), "models/Xing4.0-29B-A4B")
tok = AutoTokenizer.from_pretrained(M, trust_remote_code=True)
X = os.environ.get("XG4_WORK", "/tmp/xing4-heldout")
text = gptq.agent_texts(tok, [X + "/heldout.jsonl"])[0]
import re
parts = re.split(r"(<tool_call>.*?</tool_call>)", text, flags=re.S)
ids, mask = [], []
first_user = json.load(open(X + "/heldout.jsonl"))["messages"][1]["content"]
t0 = None; pos = 0
for part in parts:
    if not part: continue
    if t0 is None and first_user in part:
        k = part.find(first_user)
        pre = tok(part[:k], add_special_tokens=False)["input_ids"]
        t0 = len(ids) + len(pre)
    e = tok(part, add_special_tokens=False)["input_ids"]
    ids += e; mask += [1 if part.startswith("<tool_call>") else 0] * len(e)
ref = tok(text, add_special_tokens=False)["input_ids"]
print("same ids as whole-text tokenizing:", ref == ids, len(ref), len(ids))
np.array(ids, np.int32).tofile(X + "/heldout.i32"); np.array(mask, np.uint8).tofile(X + "/heldout.mask")
print("T", len(ids), "conversation from token", t0, "rows", len(ids) - t0, "call tokens", sum(mask))
