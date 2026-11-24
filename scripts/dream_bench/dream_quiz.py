#!/usr/bin/env python3
"""Run the coaching quiz (quiz/) as Dream sessions against an `ie serve` endpoint, and grade what the model leaves.

  dream_quiz.py --url http://127.0.0.1:11499 --out run.json [--ids a,b] [--parallel 3] [--think] [--temp 0.6]
                [--max-steps 0 (none)] [--max-secs 1800] [--max-tokens 6000] [--opening FILE]

Each problem is one Dream session: Dream's own system prompt and tool list (--opening, a saved Dream request: its
"system" and "tools"), then the task as the user turn, plus a line naming the file to create in the workspace. The
model works the way it does in Dream: it calls write_file / read_file / str_replace_edit / list_dir / run_bash by
name (or through Dream's tool_call wrapper). This runner carries those calls out itself in a fresh scratch workspace
(results worded like Dream's), up to --max-steps model turns. When the model stops, the file it was asked for is
graded with quiz.py's hidden tests. Fix problems start with the broken file in the workspace; test-writing problems
start with the correct implementation there (so its tests can be run against it).

The output keeps every session's whole message list (the Dream-format transcript) beside the grade, so passing
sessions can become training examples.
"""
import argparse, concurrent.futures as cf, json, os, re, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quiz  # noqa: E402

DEF_OPENING = os.path.join(os.environ.get("IE_DATA_DRIVE", os.path.expanduser("~/data")), "dream-dataset/coach/dream_opening.json")
FILE_TOOLS = {"write_file", "read_file", "str_replace_edit", "list_dir", "run_bash"}
SCHEMAS = {   # what tool_schema answers (Dream's own parameter names)
    "write_file": {"path": "string", "content": "string", "append": "boolean (optional)"},
    "read_file": {"path": "string", "offset": "int (optional, characters)", "limit": "int (optional)"},
    "str_replace_edit": {"path": "string", "old_string": "string (must occur exactly once)", "new_string": "string"},
    "list_dir": {"path": "string"},
    "run_bash": {"command": "string"},
}
MAX_OUT = 8000


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def _path(self, raw):
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("needs a 'path' argument")
        p = (self.root / raw).resolve() if not os.path.isabs(raw) else Path(raw).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError(f"{raw} is outside the workspace")
        return p

    def call(self, name, args):
        """(ok, text) for one tool call, worded like Dream's results."""
        try:
            if name == "write_file":
                p = self._path(args.get("path")); content = str(args.get("content", ""))
                p.parent.mkdir(parents=True, exist_ok=True)
                if args.get("append") is True:
                    with p.open("a") as f: f.write(content)
                    return True, f"Appended {len(content)} chars to {p}"
                p.write_text(content)
                return True, f"Wrote {len(content)} chars to {p}"
            if name == "read_file":
                p = self._path(args.get("path"))
                if not p.is_file(): return False, f"No file at {p}"
                text = p.read_text(errors="replace"); off = int(args.get("offset") or 0)
                lim = int(args.get("limit") or MAX_OUT)
                return True, text[off:off + lim]
            if name == "str_replace_edit":
                p = self._path(args.get("path"))
                if not p.is_file(): return False, f"No file at {p}"
                old, new = args.get("old_string"), args.get("new_string")
                if old is None or new is None: return False, "str_replace_edit needs both old_string and new_string."
                text = p.read_text(); n = text.count(old)
                if n != 1: return False, f"old_string occurs {n} times (it must occur exactly once). Nothing was changed."
                p.write_text(text.replace(old, new, 1))
                return True, f"Edited {p}"
            if name == "list_dir":
                p = self._path(args.get("path") or ".")
                if not p.is_dir(): return False, f"No directory at {p}"
                return True, "\n".join(sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir())) or "(empty)"
            if name == "run_bash":
                cmd = args.get("command")
                if not isinstance(cmd, str) or not cmd.strip(): return False, "run_bash needs a 'command' argument."
                home = os.path.expanduser("~")
                env = {"PATH": f"{home}/.cargo/bin:" + os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.root),
                       "LANG": "C.UTF-8", "RUSTUP_HOME": os.environ.get("RUSTUP_HOME", f"{home}/.rustup"),
                       "CARGO_HOME": os.environ.get("CARGO_HOME", f"{home}/.cargo")}
                try:
                    r = subprocess.run(["bash", "-c", cmd], cwd=self.root, capture_output=True, timeout=60, env=env)
                except subprocess.TimeoutExpired:
                    return False, "Command timed out (limit 60s); owned processes terminated."
                out = (r.stdout + r.stderr).decode("utf-8", "replace")
                if len(out) > MAX_OUT: out = out[:MAX_OUT // 2] + "\n[...output omitted...]\n" + out[-MAX_OUT // 2:]
                return r.returncode == 0, f"Command exited {r.returncode}.\n{out}"
        except Exception as e:  # noqa: BLE001 -- a tool error is reported to the model, as Dream does
            return False, f"{type(e).__name__}: {e}"
        return False, f"unknown tool {name}"


def dispatch(ws, name, raw_args):
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except json.JSONDecodeError as e:
        return False, f"The arguments are not valid JSON ({e}). Nothing was run."
    if not isinstance(args, dict):
        return False, "The arguments must be a JSON object."
    if name == "tool_call":                       # Dream's wrapper: tool_call(name, arguments)
        inner, a = args.get("name"), args.get("arguments", {})
        if isinstance(a, str):
            try: a = json.loads(a)
            except json.JSONDecodeError: return False, "tool_call: 'arguments' is not valid JSON."
        return dispatch(ws, inner, a if isinstance(a, dict) else {})
    if name == "tool_schema":
        want = args.get("name") or args.get("search") or ""
        hits = {k: v for k, v in SCHEMAS.items() if not want or want in k}
        return True, json.dumps({k: {"parameters": v} for k, v in hits.items()}) if hits else "No matching tool."
    if name in FILE_TOOLS:
        return ws.call(name, args)
    return False, f"{name} is not available in this session. Use write_file, read_file, str_replace_edit, list_dir or run_bash."


_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_FN_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
_PAR_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)


def text_tool_calls(text):
    """Dream's recovery (openai_compat._parse_text_tool_calls): complete calls written as text, in order. A reply
    with an unfinished call yields none. The server hands such calls back as text when the tool is not in the
    request's tools list -- Dream lists its catalog tools only behind tool_call, the model calls them by name."""
    if text.count("<tool_call>") != text.count("</tool_call>"):
        return []
    calls = []
    for i, m in enumerate(_CALL_RE.finditer(text)):
        body = m.group(1)
        f = _FN_RE.search(body)
        if f:
            args = {}
            for k, v in _PAR_RE.findall(f.group(2)):
                v = v[1:] if v.startswith("\n") else v
                v = v[:-1] if v.endswith("\n") else v
                if k in ("content", "command", "path", "old_string", "new_string"):
                    args[k] = v
                else:
                    try: args[k] = json.loads(v)
                    except json.JSONDecodeError: args[k] = v
            if len(args) == 1 and isinstance(next(iter(args.values())), dict) and next(iter(args)) in ("arguments", "args", "parameters"):
                args = next(iter(args.values()))
            calls.append({"id": f"text_{i}", "type": "function", "function": {"name": f.group(1), "arguments": json.dumps(args)}})
            continue
        try:
            j = json.loads(body.strip())
        except json.JSONDecodeError:
            return []                       # a block we cannot read: like Dream, run none of them
        if isinstance(j, dict) and isinstance(j.get("name"), str):
            a = j.get("arguments", {})
            calls.append({"id": f"text_{i}", "type": "function",
                          "function": {"name": j["name"], "arguments": a if isinstance(a, str) else json.dumps(a)}})
    return calls


def chat(url, body, timeout=1800):
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def run_problem(pid, opening, a):
    m = quiz.load(pid); ext = quiz.EXT[m["lang"]]
    with tempfile.TemporaryDirectory(prefix="dreamquiz-") as t:
        ws = Workspace(Path(t))
        if m["kind"] == "fix":
            (ws.root / m["file"]).write_text((m["dir"] / f"buggy{ext}").read_text())
        elif m["kind"] == "tests":
            (ws.root / f"solution{ext}").write_text((m["dir"] / f"solution{ext}").read_text())
        run_hint = {"python": f"python3 {m['file']}", "js": f"node {m['file']}",
                    "rust": "a small main.rs with `mod solution;`, built with `rustc --edition 2021 main.rs`",
                    "cpp": "a small test.cpp that includes solution.hpp, built with `g++ -std=c++20 test.cpp`"}[m["lang"]]
        user = (m["task"].rstrip() + f"\n\nWork in the current workspace (it is the working directory for run_bash). "
                f"Create `{m['file']}` there with write_file"
                + (" (the current version is already there)" if m["kind"] == "fix" else "")
                + f", run it to check your work (for example `{run_hint}`), and finish with a short summary.")
        # Dream names each session's own working directory in its prompt: point it at this scratch workspace
        system = opening["system"].replace(opening["workspace"], str(ws.root)) if opening.get("workspace") else opening["system"]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        steps = calls = bad_calls = toks = text_calls = 0; t0 = time.time(); stop = "step or time limit"
        while (a.max_steps <= 0 or steps < a.max_steps) and time.time() - t0 < a.max_secs:
            steps += 1
            body = {"model": "x", "messages": messages, "tools": opening["tools"], "temperature": a.temp, "top_p": 0.95,
                    "max_tokens": a.max_tokens, "chat_template_kwargs": {"enable_thinking": a.think}}
            try:
                r = chat(a.url, body)
            except Exception as e:  # noqa: BLE001
                stop = f"request error: {e}"[:200]; break
            msg = r["choices"][0]["message"]; toks += r.get("usage", {}).get("completion_tokens", 0)
            tcs = msg.get("tool_calls") or []
            recovered = False
            if not tcs and msg.get("content"):
                tcs = text_tool_calls(msg["content"]); recovered = bool(tcs)
            text_calls += recovered
            entry = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("reasoning_content"): entry["reasoning_content"] = msg["reasoning_content"]
            if tcs and not recovered: entry["tool_calls"] = tcs   # recovered calls stay as the text the model wrote
            messages.append(entry)
            if not tcs:
                stop = r["choices"][0].get("finish_reason") or "stop"; break
            for tc in tcs:
                calls += 1
                fn = tc.get("function", {})
                ok, text = dispatch(ws, fn.get("name"), fn.get("arguments"))
                bad_calls += not ok
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "name": fn.get("name"),
                                 "content": ("" if ok else "Error: ") + text})
        target = ws.root / m["file"]
        if target.is_file():
            g = quiz.grade_source(m, target.read_text(errors="replace"))
        else:
            g = {"pass": False, "detail": f"{m['file']} was not created"}
        return {"id": pid, "area": m["area"], "lang": m["lang"], "kind": m["kind"], "pass": g["pass"],
                "detail": g["detail"][-600:], "caught": g.get("caught"), "steps": steps, "tool_calls": calls,
                "failed_calls": bad_calls, "text_call_turns": text_calls, "completion_tokens": toks, "secs": round(time.time() - t0, 1), "stop": stop,
                "messages": messages}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:11499"); ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default=""); ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--think", action="store_true"); ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--max-steps", type=int, default=0)   # 0 = no limit, as in Dream (max_iterations=None)
    ap.add_argument("--max-secs", type=int, default=1800)  # wall-clock safety stop per problem; ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--opening", default=DEF_OPENING); ap.add_argument("--label", default="")
    a = ap.parse_args()
    opening = json.loads(Path(a.opening).read_text())
    sysmsg = opening["system"]                 # saved either as the text or as the whole {"role","content"} message
    opening["system"] = sysmsg["content"] if isinstance(sysmsg, dict) else sysmsg
    if not isinstance(opening["system"], str) or not opening["system"].strip():
        sys.exit("the opening's system prompt is not text")
    ids = a.ids.split(",") if a.ids else quiz.problems()
    t0 = time.time(); results = []
    with cf.ThreadPoolExecutor(max_workers=a.parallel) as ex:
        futs = {ex.submit(run_problem, pid, opening, a): pid for pid in ids}
        for f in cf.as_completed(futs):
            r = f.result(); results.append(r)
            print(f"{'PASS' if r['pass'] else 'fail'} {r['id']:26s} steps {r['steps']:2d} calls {r['tool_calls']:2d} "
                  f"({r['failed_calls']} failed) {r['secs']:6.0f} s | {r['stop']} | {r['detail'][:90].replace(chr(10), ' ')}",
                  flush=True)
    results.sort(key=lambda r: r["id"])
    n = len(results); p = sum(r["pass"] for r in results)
    print(f"{a.label or 'run'}: {p}/{n} passed in {time.time() - t0:.0f} s")
    json.dump({"label": a.label, "passed": p, "n": n, "think": a.think, "temp": a.temp, "results": results},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
