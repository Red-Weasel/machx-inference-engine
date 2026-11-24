#!/usr/bin/env python3
"""Inkling 2 Mini chat-template goldens: the checkpoint's own chat_template.jinja rendered by transformers'
apply_chat_template (the tokenizer of the model directory) over conversations that exercise every branch the engine's
native renderer (src/engine/inkling2_engine.cpp, inkling2_render_prompt) mirrors, plus the token ids of each rendered
prompt (tokenizer.encode, no special tokens added). Writes tests/data/inkling2_template_golden.json; the unit test
tests/unit/inkling2_serve_test.cpp renders the same cases natively and compares byte for byte (and the ids, when the
model's tokenizer.json is on disk).

    usage: <reference venv>/bin/python -I gen_template_golden.py <model dir> <out.json>

Each case holds the conversation as an OpenAI client sends it (tool-call arguments as a JSON STRING, role "developer"
allowed) -- what the server hands the engine. The template itself refuses both (it wants arguments as a parsed object,
"canonicalize upstream", and knows no developer role), so the reference is given the arguments parsed (key order kept)
and developer turns as system turns: the engine does the same before rendering. Thinking off = reasoning_effort "none"
(the template's only switch, effort 0); thinking on with no effort = reasoning_effort undefined (the template's 0.9)."""
import json
import sys

from transformers import AutoTokenizer

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather for a city — °C or °F",
    "parameters": {"type": "object", "properties": {"unit": {"type": "string", "enum": ["c", "f"]}, "city": {"type": "string"},
                                                   "days": {"type": "integer", "default": 1}, "ratio": {"type": "number", "default": 1.0}},
                   "required": ["city"]}}}
READ = {"type": "function", "function": {
    "name": "read_file", "description": "Read a file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}}}
NOOP = {"type": "function", "function": {"name": "noop"}}                       # no description, no parameters
EMPTY = {"type": "function", "function": {"name": "empty", "description": "", "parameters": {}}}   # falsy -> defaults
BARE = {"name": "bare_tool", "description": "A tool given without the function wrapper",       # tool.function undefined
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}
TYPED = {"type": "custom", "function": {"name": "typed", "description": "type kept"}}

CALL = {"id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"city": "Zürich", "days": 3, "nested": {"b": [1, 2.5, "x"], "a": None}})}}
# keys out of order at the top level (sorted, case-sensitively: "B" < "a" < "b" < "é") and inside (kept as written),
# every escape json.dumps writes, numbers that print differently in naive formatters
WILD_ARGS = '{"b": {"z": 1, "a": {"y": 2, "b": 3}}, "a": "quote \\" backslash \\\\ nl \\n tab \\t ctl \\u0001 del \\u007f </s> 東京 🙂", ' \
            '"B": [true, false, null, 0.1, 1.0, 2.5, 1e-05, -3, 1e+20], "é": {}}'
CALL_A = {"id": "call_a", "type": "function", "function": {"name": "read_file", "arguments": WILD_ARGS}}
CALL_B = {"id": "call_b", "type": "function", "function": {"name": "get_weather", "arguments": ""}}          # "" -> {}
CALL_C = {"id": "call_c", "type": "function", "function": {"name": "noop"}}                             # no arguments key
CALL_D = {"id": "call_d", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}

CASES = [
    {"name": "user only, thinking default", "thinking": True, "effort": None,
     "messages": [{"role": "user", "content": "What is the capital of France?"}]},
    {"name": "system + user, thinking off", "thinking": False, "effort": None,
     "messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}]},
    {"name": "history with reasoning, effort low", "thinking": True, "effort": "low",
     "messages": [{"role": "user", "content": "2+2?"},
                  {"role": "assistant", "content": "4", "reasoning_content": "Add them."},
                  {"role": "user", "content": "times 3?"}]},
    {"name": "tools, a call, its result by id, effort max", "thinking": True, "effort": "max",
     "messages": [{"role": "system", "content": "Use tools when needed."},
                  {"role": "user", "content": "Weather in Zürich?"},
                  {"role": "assistant", "content": "", "tool_calls": [CALL], "reasoning_content": "Call the tool."},
                  {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 18}"},
                  {"role": "assistant", "content": "18 °C.", "reasoning_content": "Answer."},
                  {"role": "user", "content": "Thanks"}],
     "tools": [WEATHER]},
    {"name": "two systems, user, mid system, effort minimal", "thinking": True, "effort": "minimal",
     "messages": [{"role": "system", "content": "First."}, {"role": "system", "content": "Second."},
                  {"role": "user", "content": "a"}, {"role": "system", "content": "Mid system."}, {"role": "user", "content": "b"}]},
    {"name": "text + call, thinking off, tool defaults", "thinking": False, "effort": None,
     "messages": [{"role": "user", "content": "x"},
                  {"role": "assistant", "content": "Checking.", "tool_calls": [CALL]},
                  {"role": "tool", "tool_call_id": "call_1", "content": "ok"}],
     "tools": [WEATHER, NOOP, EMPTY, BARE, TYPED]},
    {"name": "empty assistant turn, effort medium", "thinking": True, "effort": "medium",
     "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}, {"role": "user", "content": "again"}]},
    {"name": "system only, effort xhigh (effort last)", "thinking": True, "effort": "xhigh",
     "messages": [{"role": "system", "content": "Only a system message."}]},
    {"name": "parallel calls, wild args, effort none", "thinking": True, "effort": "none",
     "messages": [{"role": "user", "content": "Read it and check the weather."},
                  {"role": "assistant", "content": "", "reasoning_content": "Two calls.", "tool_calls": [CALL_A, CALL_B, CALL_C]},
                  {"role": "tool", "tool_call_id": "call_a", "content": "file body"},
                  {"role": "tool", "tool_call_id": "call_b", "content": "sunny"},
                  {"role": "tool", "tool_call_id": "call_c", "content": ""}],
     "tools": [READ, WEATHER, NOOP]},
    {"name": "result without id, empty user, effort high", "thinking": True, "effort": "high",
     "messages": [{"role": "user", "content": ""},
                  {"role": "assistant", "content": "", "tool_calls": [CALL_D]},
                  {"role": "tool", "content": "orphan result"},
                  {"role": "tool", "tool_call_id": "nope", "content": "unmatched id"}],
     "tools": [READ]},
    {"name": "developer turn as system", "thinking": True, "effort": None,
     "messages": [{"role": "developer", "content": "Dev note."}, {"role": "user", "content": "Zürich — 東京 🙂\nline 2"}]},
    {"name": "reasoning-only assistant turn", "thinking": True, "effort": "low",
     "messages": [{"role": "user", "content": "hm"}, {"role": "assistant", "content": "", "reasoning_content": "Only thoughts."},
                  {"role": "user", "content": "go on"}]},
    {"name": "agent loop, two rounds, thinking off", "thinking": False, "effort": None,
     "messages": [{"role": "system", "content": "Agent."}, {"role": "user", "content": "Find it."},
                  {"role": "assistant", "content": "", "tool_calls": [CALL_D]},
                  {"role": "tool", "tool_call_id": "call_d", "content": "a.txt\nb.txt"},
                  {"role": "assistant", "content": "Reading a.txt.", "tool_calls": [CALL_A]},
                  {"role": "tool", "tool_call_id": "call_a", "content": "hello"},
                  {"role": "assistant", "content": "It says hello."},
                  {"role": "user", "content": "ok"}],
     "tools": [READ]},
]


def for_reference(messages):
    out = []
    for m in messages:
        m = json.loads(json.dumps(m))
        if m["role"] == "developer":
            m["role"] = "system"
        for tc in m.get("tool_calls", []):
            a = tc["function"].get("arguments")
            if isinstance(a, str):
                tc["function"]["arguments"] = json.loads(a) if a.strip() else {}
        out.append(m)
    return out


def main():
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    out = []
    for c in CASES:
        kw = {}
        if c.get("tools"):
            kw["tools"] = c["tools"]
        if not c["thinking"]:
            kw["reasoning_effort"] = "none"
        elif c["effort"] is not None:
            kw["reasoning_effort"] = c["effort"]
        prompt = tok.apply_chat_template(for_reference(c["messages"]), tokenize=False, **kw)
        gen = tok.apply_chat_template(for_reference(c["messages"]), tokenize=False, add_generation_prompt=True, **kw)
        assert gen == prompt, "the template grew a generation-prompt branch"
        ids = tok.encode(prompt, add_special_tokens=False)
        out.append({**c, "prompt": prompt, "ids": ids})
    json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(out)} cases -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
