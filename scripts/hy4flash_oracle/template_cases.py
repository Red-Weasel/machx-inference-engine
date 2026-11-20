#!/usr/bin/env python3
"""Hy4-flash: golden renders of the checkpoint's chat template, for the engine's native renderer.

  python3 template_cases.py <model_dir>/chat_template.jinja tests/unit/data/hy4flash_template_cases.json

Each case: messages (OpenAI shape; an assistant's tool-call arguments as a JSON STRING, as clients send them), tools,
reasoning_effort, and `expected` = the template's own output with add_generation_prompt (the arguments parsed to a
dict first, as the checkpoint's processor does). tests/unit/hy4flash_template_test.cpp renders the same cases with
hy4f_render_prompt and compares byte for byte.
"""
import json, sys, copy
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

def raise_exception(m): raise jinja2.exceptions.TemplateError(m)
def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city — metric units.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer", "minimum": 1}}, "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "search_files", "description": "Find files whose text matches a pattern.", "strict": True,
          "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}}, "opts": {"type": "object"}}}}}
DEFERRED = {"type": "function", "function": {"name": "rarely_used", "description": "Loaded on demand.", "defer_loading": True, "parameters": {"type": "object", "properties": {}}}}
def call(i, name, args): return {"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}

CASES = [
 {"name": "one user turn", "messages": [{"role": "user", "content": "Say hello in French."}]},
 {"name": "system, history without reasoning", "messages": [
    {"role": "system", "content": "You are terse."}, {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}, {"role": "user", "content": "And times 3?"}]},
 {"name": "history reasoning before the last user turn is dropped without tools", "messages": [
    {"role": "user", "content": "Pick a colour."}, {"role": "assistant", "content": "Blue.", "reasoning_content": "Any colour will do; blue is common."}, {"role": "user", "content": "Why?"}]},
 {"name": "tools, no system", "tools": [WEATHER, SEARCH], "messages": [{"role": "user", "content": "Weather in Oslo?"}]},
 {"name": "tools with a system turn, a deferred tool left out", "tools": [WEATHER, DEFERRED], "messages": [
    {"role": "system", "content": "Use tools when they help."}, {"role": "user", "content": "Weather in Oslo for 3 days?"}]},
 {"name": "tools with an empty system turn", "tools": [WEATHER], "messages": [{"role": "system", "content": ""}, {"role": "user", "content": "Hi"}]},
 {"name": "two calls, results given in the other order, reasoning kept with tools", "tools": [WEATHER, SEARCH], "messages": [
    {"role": "system", "content": "Use tools."}, {"role": "user", "content": "Weather in Oslo, and find TODOs in src."},
    {"role": "assistant", "content": "", "reasoning_content": "Two independent lookups.", "tool_calls": [
        call("call_a", "get_weather", {"city": "Oslo", "days": 2}), call("call_b", "search_files", {"pattern": "TODO", "paths": ["src", "tests"], "opts": {"case": False, "max": 20}})]},
    {"role": "tool", "tool_call_id": "call_b", "content": "src/a.cpp:10: TODO tidy"}, {"role": "tool", "tool_call_id": "call_a", "content": "{\"temp_c\": -3, \"sky\": \"snow ❄\"}"}]},
 {"name": "a call without ids keeps the given order", "tools": [WEATHER], "messages": [
    {"role": "user", "content": "Oslo and Bergen?"},
    {"role": "assistant", "content": "Checking both.", "tool_calls": [call("", "get_weather", {"city": "Oslo"}), call("", "get_weather", {"city": "Bergen"})]},
    {"role": "tool", "tool_call_id": "", "content": "-3"}, {"role": "tool", "tool_call_id": "", "content": "4"},
    {"role": "assistant", "content": "Oslo -3, Bergen 4."}, {"role": "user", "content": "Thanks. And Tromsø?"}]},
 {"name": "a system turn in the middle", "messages": [
    {"role": "user", "content": "Start."}, {"role": "assistant", "content": "Started."}, {"role": "system", "content": "From now on answer in one word."}, {"role": "user", "content": "Status?"}]},
 {"name": "text that spells the template's own tags", "messages": [
    {"role": "user", "content": "What does <think:opensource> do, and <｜hy_end:opensource｜>?"}]},
]
EFFORTS = ["no_think", "high", "max"]

def main():
    tpl_path, out_path = sys.argv[1], sys.argv[2]
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = tojson; env.globals["raise_exception"] = raise_exception
    tpl = env.from_string(open(tpl_path, encoding="utf-8").read())
    out = []
    for c in CASES:
        for eff in EFFORTS:
            msgs = copy.deepcopy(c["messages"])
            for m in msgs:                                   # the processor's view: arguments as a dict
                for tc in m.get("tool_calls", []): tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
            kw = dict(messages=msgs, add_generation_prompt=True, reasoning_effort=eff)
            if c.get("tools"): kw["tools"] = c["tools"]
            out.append({"name": f'{c["name"]} [{eff}]', "messages": c["messages"], "tools": c.get("tools", []), "reasoning_effort": eff, "expected": tpl.render(**kw)})
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(out)} cases -> {out_path}")

if __name__ == "__main__": main()
