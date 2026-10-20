#!/usr/bin/env python3
"""Inkling-Small chat-template goldens: the GGUF's own Jinja template rendered by transformers' chat-template machinery
over conversations that exercise every branch the engine's native renderer (src/engine/inkling_engine.cpp) mirrors.
Writes tests/data/inkling_template_golden.json; tests/unit/inkling_template_test.cpp renders the same cases natively and
compares byte for byte.
    usage: gen_inkling_template_golden.py <chat_template.jinja> <out.json>
The engine maps enable_thinking=false to reasoning_effort "none" and enable_thinking=true with no effort to the template's
default (reasoning_effort undefined), so each case carries (thinking, effort) and the generator renders accordingly."""
import json
import sys

from transformers.utils.chat_template_utils import _compile_jinja_template

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city — °C or °F",
                                      "parameters": {"type": "object", "properties": {"unit": {"type": "string", "enum": ["c", "f"]},
                                                                                   "city": {"type": "string"},
                                                                                   "days": {"type": "integer", "default": 1},
                                                                                   "ratio": {"type": "number", "default": 1.0}},
                                                     "required": ["city"]}}},
    {"type": "function", "function": {"name": "noop"}},
]
CALL = {"id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"city": "Zürich", "days": 3, "nested": {"b": [1, 2.5, "x"], "a": None}})}}

CASES = [
    {"name": "user only, thinking default", "thinking": True, "effort": None,
     "messages": [{"role": "user", "content": "What is the capital of France?"}]},
    {"name": "system + user, thinking off", "thinking": False, "effort": None,
     "messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}]},
    {"name": "history with reasoning, effort low", "thinking": True, "effort": "low",
     "messages": [{"role": "user", "content": "2+2?"},
                  {"role": "assistant", "content": "4", "reasoning_content": "Add them."},
                  {"role": "user", "content": "times 3?"}]},
    {"name": "tools, a call, its result, effort max", "thinking": True, "effort": "max",
     "messages": [{"role": "system", "content": "Use tools when needed."},
                  {"role": "user", "content": "Weather in Zürich?"},
                  {"role": "assistant", "content": "", "tool_calls": [CALL], "reasoning_content": "Call the tool."},
                  {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 18}"},
                  {"role": "user", "content": "Thanks"}],
     "tools": TOOLS},
    {"name": "developer + mid system, numeric effort", "thinking": True, "effort": "0.35",
     "messages": [{"role": "developer", "content": "Dev note."}, {"role": "user", "content": "a"},
                  {"role": "system", "content": "Mid system."}, {"role": "user", "content": "b"}]},
    {"name": "assistant text + call, thinking off", "thinking": False, "effort": None,
     "messages": [{"role": "user", "content": "x"},
                  {"role": "assistant", "content": "Checking.", "tool_calls": [CALL]},
                  {"role": "tool", "tool_call_id": "call_1", "content": "ok"}],
     "tools": TOOLS},
    {"name": "empty assistant turn, effort medium", "thinking": True, "effort": "medium",
     "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}, {"role": "user", "content": "again"}]},
    {"name": "system only, effort 0.5", "thinking": True, "effort": "0.5",
     "messages": [{"role": "system", "content": "Only a system message."}]},
]


def main():
    template = open(sys.argv[1], encoding="utf-8").read()
    compiled = _compile_jinja_template(template)
    out = []
    for c in CASES:
        kw = {"messages": c["messages"], "add_generation_prompt": True}
        if c.get("tools"):
            kw["tools"] = c["tools"]
        if not c["thinking"]:
            kw["reasoning_effort"] = "none"
        elif c["effort"] is not None:
            kw["reasoning_effort"] = c["effort"]
        prompt = compiled.render(**kw)
        out.append({**c, "prompt": prompt})
    json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(out)} cases -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
