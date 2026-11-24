#!/usr/bin/env python3
"""Tool-call benchmark through ie serve: 12 agent tools with typed parameters (integers, numbers, enums, arrays, objects),
36 requests with a known right call, each sent N times at the model's sampling temperature. Per reply:
  valid   = a tool call that parses, names a real tool, has every required key, no unknown key, every value of its type;
  correct = valid, the expected tool, and the expected arguments (case-insensitive substring match on strings);
plus first-token time and speed. Usage: toolcall_bench.py --url http://127.0.0.1:11499 --repeats 3 --temp 1.0 --out r.json"""
import argparse, json, threading, time, urllib.request

def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": req}}}
S = {"type": "string"}; I = {"type": "integer"}; N = {"type": "number"}; B = {"type": "boolean"}
TOOLS = [
    fn("run_bash", "Run a shell command.", {"command": S, "timeout_s": I}, ["command"]),
    fn("read_file", "Read lines of a file.", {"path": S, "start_line": I, "end_line": I}, ["path"]),
    fn("write_file", "Write a whole file.", {"path": S, "content": S, "overwrite": B}, ["path", "content"]),
    fn("edit_file", "Replace exact text in a file.", {"path": S, "old_text": S, "new_text": S}, ["path", "old_text", "new_text"]),
    fn("search_code", "Search the repository with a regex.", {"pattern": S, "glob": S, "max_results": I}, ["pattern"]),
    fn("web_search", "Search the web.", {"query": S, "max_results": I, "recency": {"type": "string", "enum": ["day", "week", "month", "any"]}}, ["query"]),
    fn("git", "Run a git operation.", {"op": {"type": "string", "enum": ["status", "diff", "log", "commit"]}, "message": S, "limit": I}, ["op"]),
    fn("set_reminder", "Set a reminder.", {"when": S, "text": S, "priority": {"type": "string", "enum": ["low", "normal", "high"]}}, ["when", "text"]),
    fn("http_request", "Send an HTTP request.", {"method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]}, "url": S, "headers": {"type": "object"}, "body": S}, ["method", "url"]),
    fn("calculator", "Evaluate an arithmetic expression.", {"expression": S}, ["expression"]),
    fn("create_task", "Create a task in the tracker.", {"title": S, "tags": {"type": "array", "items": S}, "estimate_hours": N, "due": S}, ["title"]),
    fn("send_message", "Send a chat message.", {"channel": {"type": "string", "enum": ["general", "dev", "alerts"]}, "text": S, "urgent": B}, ["channel", "text"]),
]
# (prompt, expected tool, expected args: value or None = only present)
CASES = [
    ("Show me the git status of this repo.", "git", {"op": "status"}),
    ("Show the last 5 commits.", "git", {"op": "log", "limit": 5}),
    ("Commit the staged changes with the message 'fix lane reuse'.", "git", {"op": "commit", "message": "fix lane reuse"}),
    ("Read lines 40 to 80 of src/main.cpp.", "read_file", {"path": "src/main.cpp", "start_line": 40, "end_line": 80}),
    ("Open README.md.", "read_file", {"path": "README.md"}),
    ("Run the unit tests with `ctest --output-on-failure`, give it 600 seconds.", "run_bash", {"command": "ctest", "timeout_s": 600}),
    ("How much disk space is free? Use a shell command.", "run_bash", {"command": "df"}),
    ("Find every place the code calls malloc_device, at most 20 results.", "search_code", {"pattern": "malloc_device", "max_results": 20}),
    ("Search the .hpp files for 'struct LanesRequest'.", "search_code", {"pattern": "LanesRequest", "glob": "hpp"}),
    ("Look up news about Intel Arc B70 drivers from the past week.", "web_search", {"query": "Arc", "recency": "week"}),
    ("Search the web for 'GPTQ vs AWQ', 3 results.", "web_search", {"query": "GPTQ", "max_results": 3}),
    ("Remind me tomorrow at 9am to call the bank. It's important.", "set_reminder", {"when": "9", "text": "bank", "priority": "high"}),
    ("Set a low priority reminder for Friday: water the plants.", "set_reminder", {"when": "Friday", "text": "plants", "priority": "low"}),
    ("GET https://api.example.com/v1/status", "http_request", {"method": "GET", "url": "https://api.example.com/v1/status"}),
    ("POST {\"name\": \"x\"} to https://api.example.com/items with header Content-Type: application/json.", "http_request", {"method": "POST", "url": "https://api.example.com/items", "headers": None, "body": "name"}),
    ("Delete the resource at https://api.example.com/items/42.", "http_request", {"method": "DELETE", "url": "https://api.example.com/items/42"}),
    ("What is 17.5% of 2480? Use the calculator.", "calculator", {"expression": "2480"}),
    ("Compute (1024*1024*3)/7 with the calculator.", "calculator", {"expression": "1024"}),
    ("Create a task 'Port K2 Horizon' tagged engine and port, estimate 12.5 hours.", "create_task", {"title": "K2 Horizon", "tags": None, "estimate_hours": 12.5}),
    ("Add a task to write the release notes, due Monday.", "create_task", {"title": "release notes", "due": "Monday"}),
    ("Tell the dev channel that the build is green.", "send_message", {"channel": "dev", "text": "green"}),
    ("Send an urgent message to alerts: card 0 is overheating.", "send_message", {"channel": "alerts", "text": "overheating", "urgent": True}),
    ("Post 'lunch at noon' in general.", "send_message", {"channel": "general", "text": "lunch"}),
    ("Create notes.txt containing 'hello world', overwrite it if it exists.", "write_file", {"path": "notes.txt", "content": "hello world", "overwrite": True}),
    ("Write a file config/app.json with {\"debug\": true}.", "write_file", {"path": "config/app.json", "content": "debug"}),
    ("In src/a.cpp replace 'int x = 0;' with 'int x = 1;'.", "edit_file", {"path": "src/a.cpp", "old_text": "int x = 0;", "new_text": "int x = 1;"}),
    ("Rename the function foo to bar in lib/util.py (it appears as 'def foo(').", "edit_file", {"path": "lib/util.py", "old_text": "def foo(", "new_text": "def bar("}),
    ("List the files in the current directory.", "run_bash", {"command": "ls"}),
    ("Show me what changed in the working tree.", "git", {"op": "diff"}),
    ("Read the first 10 lines of /etc/os-release.", "read_file", {"path": "/etc/os-release", "end_line": 10}),
    ("Search for TODO comments in the C++ sources.", "search_code", {"pattern": "TODO"}),
    ("Find recent (today) reports about the DeepSeek V4 release.", "web_search", {"query": "DeepSeek", "recency": "day"}),
    ("PUT 'enabled' to https://api.example.com/flags/7.", "http_request", {"method": "PUT", "url": "https://api.example.com/flags/7", "body": "enabled"}),
    ("Create a task 'Profile Xing4 lanes' with no tags.", "create_task", {"title": "Profile Xing4"}),
    ("Check the GPU temperatures with xpu-smi, time out after 30 seconds.", "run_bash", {"command": "xpu-smi", "timeout_s": 30}),
    ("Remind me in 2 hours to check the GPTQ run, normal priority.", "set_reminder", {"when": "2", "text": "GPTQ", "priority": "normal"}),
]
TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}
SPEC = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}

def check(call, want_tool, want_args):
    f = call.get("function", {})
    name = f.get("name")
    if name not in SPEC: return False, False, f"unknown tool {name!r}"
    try: args = json.loads(f.get("arguments") or "{}") if isinstance(f.get("arguments"), str) else f.get("arguments")
    except ValueError: return False, False, "arguments not JSON"
    if not isinstance(args, dict): return False, False, "arguments not an object"
    ps = SPEC[name]
    for r in ps["required"]:
        if r not in args: return False, False, f"missing {r}"
    for k, v in args.items():
        if k not in ps["properties"]: return False, False, f"unknown key {k}"
        p = ps["properties"][k]; ty = TYPES[p["type"]]
        if isinstance(v, bool) and p["type"] in ("integer", "number"): return False, False, f"{k} bool for {p['type']}"
        if not isinstance(v, ty): return False, False, f"{k} not {p['type']}"
        if "enum" in p and v not in p["enum"]: return False, False, f"{k}={v!r} not in enum"
    if name != want_tool: return True, False, f"wrong tool {name}"
    for k, w in want_args.items():
        if k not in args: return True, False, f"no {k}"
        if w is None: continue
        v = args[k]
        if isinstance(w, str):
            if not isinstance(v, str) or w.lower() not in v.lower(): return True, False, f"{k}={v!r}"
        elif v != w: return True, False, f"{k}={v!r}"
    return True, True, ""

NO_GUARD = False


def run(url, case, temp, think, results, i):
    prompt, tool, args = case
    body = {"model": "x", "messages": [{"role": "system", "content": "You are an agent. Use exactly one tool call to do what the user asks."},
                                        {"role": "user", "content": prompt}],
            "tools": TOOLS, "max_tokens": 1024, "temperature": temp, "top_p": 0.95, "stream": True,
            "chat_template_kwargs": {"enable_thinking": think}}
    if NO_GUARD: body["ie_tool_guard"] = False   # the engine's tool-call guard off for this request (A/B on one load)
    t0 = time.time(); first = None; n = 0; calls = {}; err = ""
    try:
        r = urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900)
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            ch = d.get("choices") or [{}]
            delta = ch[0].get("delta", {})
            if (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")) and first is None: first = time.time() - t0
            if delta.get("content") or delta.get("reasoning_content"): n += 1
            for tc in delta.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"function": {"name": "", "arguments": ""}})
                f = tc.get("function", {})
                if f.get("name"): c["function"]["name"] += f["name"]
                if f.get("arguments"): c["function"]["arguments"] += f["arguments"]
    except Exception as e: err = f"request failed: {e}"
    dt = time.time() - t0
    if err: v, ok, why = False, False, err
    elif not calls: v, ok, why = False, False, "no tool call"
    else: v, ok, why = check(calls[min(calls)], tool, args)
    results[i] = {"prompt": prompt, "valid": v, "correct": ok, "why": why, "ttft": first, "secs": dt,
                  "call": calls.get(min(calls)) if calls else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11499"); ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--think", action="store_true")
    ap.add_argument("--parallel", type=int, default=16); ap.add_argument("--out", default=""); ap.add_argument("--label", default="")
    ap.add_argument("--no-guard", action="store_true", help="send ie_tool_guard false (the engine's guard off per request)")
    a = ap.parse_args()
    global NO_GUARD; NO_GUARD = a.no_guard
    jobs = [c for c in CASES for _ in range(a.repeats)]
    results = [None] * len(jobs); sem = threading.Semaphore(a.parallel)
    def go(i):
        with sem: run(a.url, jobs[i], a.temp, a.think, results, i)
    t0 = time.time(); th = [threading.Thread(target=go, args=(i,)) for i in range(len(jobs))]
    [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
    nv = sum(r["valid"] for r in results); nc = sum(r["correct"] for r in results); n = len(results)
    tt = sorted(r["ttft"] for r in results if r["ttft"] is not None)
    print(f"{a.label}: {n} requests ({len(CASES)} cases x {a.repeats}, temp {a.temp}, thinking {'on' if a.think else 'off'}) in {wall:.0f} s")
    print(f"  valid tool calls   {nv}/{n} = {100 * nv / n:.1f} %")
    print(f"  correct calls      {nc}/{n} = {100 * nc / n:.1f} %")
    if tt: print(f"  first token        median {tt[len(tt) // 2]:.2f} s")
    from collections import Counter
    bad = Counter(r["why"].split(" ")[0] + " " + " ".join(r["why"].split(" ")[1:2]) for r in results if not r["valid"])
    if bad: print("  invalid because    " + "; ".join(f"{k.strip()} x{v}" for k, v in bad.most_common(6)))
    wrong = Counter(r["why"] for r in results if r["valid"] and not r["correct"])
    if wrong: print("  valid but wrong    " + "; ".join(f"{k} x{v}" for k, v in wrong.most_common(6)))
    if a.out: json.dump({"label": a.label, "valid": nv, "correct": nc, "n": n, "results": results}, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
