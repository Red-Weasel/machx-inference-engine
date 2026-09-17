#!/usr/bin/env python3
"""Prompt-format goldens for DeepSeek-V4.1 text conversations (docs/deepseek41/31, Phase 12 step 2),
through the checkpoint's own encoding/encoding.py `encode_messages`. For each case k:
  <out>/prompt_<k>.json   the case (messages, thinking_mode, reasoning_effort, drop_thinking)
  <out>/prompt_<k>.txt    the reference's prompt string
  <out>/prompt_<k>.i32    that string tokenised (HF tokenizers over tokenizer.json), int32 LE
Usage: prompt_golden.py <model_dir> <golden_dir> (needs the venv with tokenizers)"""
import json, os, struct, sys
model, out = sys.argv[1:3]
sys.path.insert(0, os.path.join(model, "encoding"))
from encoding import encode_messages   # noqa: E402
from tokenizers import Tokenizer       # noqa: E402
tok = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))

S = {"role": "system", "content": "You are a helpful assistant."}
U1 = {"role": "user", "content": "What is the capital of France?"}
A1 = {"role": "assistant", "content": "The capital of France is Paris.", "reasoning_content": "The user asks a geography question. Paris."}
U2 = {"role": "user", "content": "And of Germany?"}
A2 = {"role": "assistant", "content": "Berlin.", "reasoning_content": "Germany's capital is Berlin."}
U3 = {"role": "user", "content": "Thanks. Now list both with their populations, one per line.\n\nUse numbers with thousands separators."}
cases = [
    dict(name="chat_single",        messages=[U1], thinking_mode="chat"),
    dict(name="think_single",       messages=[U1], thinking_mode="thinking"),
    dict(name="think_effort30",     messages=[S, U1], thinking_mode="thinking", reasoning_effort=30),
    dict(name="chat_system",        messages=[S, U1], thinking_mode="chat"),
    dict(name="think_multi_drop",   messages=[S, U1, A1, U2, A2, U3], thinking_mode="thinking"),
    dict(name="think_multi_keep",   messages=[S, U1, A1, U2, A2, U3], thinking_mode="thinking", drop_thinking=False),
    dict(name="chat_multi",         messages=[S, U1, A1, U2, A2, U3], thinking_mode="chat"),
    dict(name="mid_system",         messages=[S, U1, A1, {"role": "system", "content": "Answer in French from now on."}, U2, A2, U3], thinking_mode="thinking"),
    dict(name="mid_system_last",    messages=[U1, A1, {"role": "system", "content": "Switch to German."}], thinking_mode="thinking"),
    dict(name="effort_low",         messages=[U1], thinking_mode="thinking", reasoning_effort="low"),
    dict(name="effort_max",         messages=[S, U1], thinking_mode="thinking", reasoning_effort="max"),
    dict(name="assistant_last",     messages=[U1, {"role": "assistant", "content": "The capital of France is", "reasoning_content": "Geography."}], thinking_mode="thinking"),
    dict(name="empty_system",       messages=[{"role": "system", "content": ""}, U1], thinking_mode="thinking"),
    dict(name="no_bos",             messages=[S, U1], thinking_mode="chat", add_default_bos_token=False),
    dict(name="unicode",            messages=[{"role": "user", "content": "Übersetze: 深度求索 🚀 — \"quoted\" & <tags>"}], thinking_mode="chat"),
]
for k, c in enumerate(cases):
    kw = {kk: c[kk] for kk in ("reasoning_effort", "drop_thinking", "add_default_bos_token") if kk in c}
    prompt = encode_messages(c["messages"], thinking_mode=c["thinking_mode"], **kw)
    ids = tok.encode(prompt, add_special_tokens=True).ids
    with open(os.path.join(out, "prompt_%d.json" % k), "w", encoding="utf-8") as f:
        json.dump({"name": c["name"], "messages": c["messages"], "thinking_mode": c["thinking_mode"],
                   "reasoning_effort": c.get("reasoning_effort"), "drop_thinking": c.get("drop_thinking", True),
                   "add_default_bos_token": c.get("add_default_bos_token", True)}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "prompt_%d.txt" % k), "w", encoding="utf-8", newline="") as f: f.write(prompt)
    with open(os.path.join(out, "prompt_%d.i32" % k), "wb") as f: f.write(struct.pack("<%di" % len(ids), *ids))
    print("%2d %-18s %5d chars %4d ids  %s" % (k, c["name"], len(prompt), len(ids), repr(prompt[:70])))
print(len(cases), "cases")
