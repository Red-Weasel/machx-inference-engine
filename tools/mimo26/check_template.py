#!/usr/bin/env python3
"""P3b contract (docs/mimo26/00_PORT_PLAN.md): the engine's C++ MiMo-V2.6 chat template vs the checkpoint's own Jinja
template (transformers apply_chat_template on chat_template.jinja), byte for byte, over a set of conversations; and the
completion parser on the model's own XML tool-call form.

  check_template.py <model_dir> <ie-mimo26-chat-render binary>

The engine primes "<think>" after the generation prompt in thinking mode (the model's own first token); that suffix is
the one expected difference and is stripped before comparing. OpenAI-format tool calls (arguments as JSON strings) go
to the engine; the SAME calls with arguments as dicts go to Jinja -- the form the model is trained on.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile

from transformers import PreTrainedTokenizerFast

model, binary = sys.argv[1], sys.argv[2]
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model, "tokenizer.json"))   # no remote code: only the template matters
tmpl = open(os.path.join(model, "chat_template.jinja"), encoding="utf-8").read()

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file. Paths are relative; \"quotes\" and ünïcode ok.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"},
                                                     "flags": {"type": "array", "items": {"type": "string"}}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_bash", "description": "Run a shell command.\nMultiline description.",
                                      "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "number"},
                                                     "background": {"type": "boolean"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "media_read", "description": "Inspect workspace media status.",
                                      "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["status", "projects", "get", "jobs"]}},
                                                     "required": ["action"]}}},
]


def call(name, args, cid="call_1"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


CASES = [
    {"name": "plain", "thinking": True, "messages": [{"role": "user", "content": "Hello!"}]},
    {"name": "plain no-think", "thinking": False, "messages": [{"role": "user", "content": "Hello!"}]},
    {"name": "system + multi-turn", "thinking": True, "messages": [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "Simple arithmetic."},
        {"role": "user", "content": "And 3+3?\nThink briefly."}]},
    {"name": "tools + call + result", "thinking": True, "tools": TOOLS, "messages": [
        {"role": "system", "content": "Agent mode."},
        {"role": "user", "content": "Show me main.cpp"},
        {"role": "assistant", "content": "Reading it.", "reasoning_content": "Need the file.",
         "tool_calls": [call("read_file", {"path": "src/main.cpp", "start_line": 1, "flags": ["a", "b"]})]},
        {"role": "tool", "tool_call_id": "call_1", "content": "int main() { return 0; }"},
        {"role": "user", "content": "Now run it."}]},
    {"name": "two calls, no content", "thinking": False, "tools": TOOLS, "messages": [
        {"role": "user", "content": "do both"},
        {"role": "assistant", "content": "", "tool_calls": [call("run_bash", {"command": "ls -la", "timeout": 30, "background": False}, "c1"),
                                                            call("read_file", {"path": "ünï/x.txt"}, "c2")]},
        {"role": "tool", "tool_call_id": "c1", "content": "total 0"},
        {"role": "tool", "tool_call_id": "c2", "content": "x"}]},
    {"name": "unicode + json-ish content", "thinking": True, "messages": [
        {"role": "user", "content": "日本語で答えて。{\"k\": [1, 2]} <b>tags</b>\t\ttabs"}]},
]

fails = 0
for c in CASES:
    msgs_engine = c["messages"]
    msgs_jinja = copy.deepcopy(c["messages"])
    for m in msgs_jinja:
        for tc in m.get("tool_calls", []):
            tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    ref = tok.apply_chat_template(msgs_jinja, tools=c.get("tools"), tokenize=False, add_generation_prompt=True,
                                  enable_thinking=c["thinking"], chat_template=tmpl)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump({"messages": msgs_engine, "thinking": c["thinking"], **({"tools": c["tools"]} if "tools" in c else {})}, f, ensure_ascii=False)
        path = f.name
    out = subprocess.run([binary, "render", path], capture_output=True).stdout.decode("utf-8")
    os.unlink(path)
    eng = out[:-len("<think>")] if c["thinking"] and out.endswith("<think>") else out
    ok = eng == ref
    fails += not ok
    print(f"{'ok ' if ok else 'DIFF'} render: {c['name']}")
    if not ok:
        i = next((k for k in range(min(len(eng), len(ref))) if eng[k] != ref[k]), min(len(eng), len(ref)))
        print(f"   first difference at char {i}:\n   jinja ={ref[max(0, i - 60):i + 80]!r}\n   engine={eng[max(0, i - 60):i + 80]!r}")

PARSE = [
    ("reasoning + content", True, "Let me think.</think>\n\nThe answer is 4.", {"reasoning": "Let me think.", "content": "\n\nThe answer is 4.", "calls": []}),
    ("one call, typed", True, "Need a file.</think>Reading.\n<tool_call><function=read_file><parameter=path>\nsrc/a.cpp\n</parameter><parameter=start_line>\n12\n</parameter><parameter=flags>[\"x\"]</parameter></function></tool_call>",
     {"reasoning": "Need a file.", "content": "Reading.", "calls": [("read_file", {"path": "src/a.cpp", "start_line": 12, "flags": ["x"]})]}),
    ("two calls, no think", False, "<tool_call><function=run_bash><parameter=command>ls</parameter><parameter=background>false</parameter></function></tool_call>\n<tool_call><function=read_file><parameter=path>12</parameter></function></tool_call>",
     {"reasoning": "", "content": "", "calls": [("run_bash", {"command": "ls", "background": False}), ("read_file", {"path": "12"})]}),
    ("cut call stays text", True, "x</think>ok<tool_call><function=read_file><parameter=path>a", {"reasoning": "x", "content": "ok<tool_call><function=read_file><parameter=path>a", "calls": []}),
    ("cut inside reasoning", True, "still thinking", {"reasoning": "still thinking", "content": "", "calls": []}),
    # Dream 2026-09-22, the uncensored checkpoint at temperature 1: a parameter with no </parameter> and a stray '">' --
    # repaired, and the well-formed calls after it are kept
    ("unterminated parameter repaired", True, "x</think><tool_call><function=read_file><parameter=path>a\"></function></tool_call>"
     "<tool_call><function=run_bash><parameter=command>ls</parameter></function></tool_call>",
     {"reasoning": "x", "content": "", "calls": [("read_file", {"path": "a"}), ("run_bash", {"command": "ls"})]}),
    ("enum value with a stray quote snapped", False, "<tool_call><function=media_read><parameter=action>status\"</parameter></function></tool_call>",
     {"reasoning": "", "content": "", "calls": [("media_read", {"action": "status"})]}),
    ("non-enum value kept as written", False, "<tool_call><function=read_file><parameter=path>a\"b</parameter></function></tool_call>",
     {"reasoning": "", "content": "", "calls": [("read_file", {"path": "a\"b"})]}),
    ("unparseable call kept as text", False, "<tool_call><function=read_file><parameter=</function></tool_call>"
     "<tool_call><function=run_bash><parameter=command>ls</parameter></function></tool_call>",
     {"reasoning": "", "content": "<tool_call><function=read_file><parameter=</function></tool_call>", "calls": [("run_bash", {"command": "ls"})]}),
]
for name, thinking, text, want in PARSE:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump({"thinking": thinking, "completion": text, "tools": TOOLS}, f, ensure_ascii=False)
        path = f.name
    got = json.loads(subprocess.run([binary, "parse", path], capture_output=True).stdout.decode("utf-8"))
    os.unlink(path)
    calls = [(tc["function"]["name"], json.loads(tc["function"]["arguments"])) for tc in got["tool_calls"]]
    ok = got["reasoning"] == want["reasoning"] and got["content"] == want["content"] and calls == want["calls"]
    fails += not ok
    print(f"{'ok ' if ok else 'DIFF'} parse: {name}" + ("" if ok else f"\n   got {got}"))
print("FAIL" if fails else "PASS", f"({fails} failing)")
sys.exit(1 if fails else 0)
