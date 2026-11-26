"""Rewards for machx-train's GRPO mode: score one reply to one task, 0.0 .. 1.0.

A GRPO task line is {"messages": [...], "reward": {...}} (+ optional "id", "group"). Only "messages" is sent to the
model; the reward spec (tests, expected answers) never is. Reward types:

  {"type": "tests", "language": "python", "answer_file": "solution.py", "files": {"test_solution.py": "..."},
   "check": [["python3", "test_solution.py"]], "timeout": 60, "extract": "fence"}
      The answer is taken from the reply (extract: "fence" = the last fenced code block, preferring one tagged with the
      language; "answer_line" = the last line starting with "ANSWER:"; "all" = the whole reply), written to answer_file
      beside `files` in a fresh temporary folder, and every `check` command must exit 0: reward 1, else 0.
  {"type": "exact", "answer": "ANSWER: Cy", "extract": "answer_line"}      1 if the extracted text equals `answer`
  {"type": "regex", "pattern": "...", "extract": "all"}                     1 if the pattern matches (re.search)
  {"type": "command", "argv": ["python3", "/path/score.py"], "timeout": 60}
      The command gets {"task": <the task line>, "reply": {"content", "reasoning", "finish_reason"}} on stdin and
      prints the reward (a number, clipped to 0..1) as its last stdout line.

Every command runs in its own systemd scope capped at 4 GiB with no swap (model-written code once ate 98 GiB on this
machine), with a wall-clock limit. A reply cut off at the token limit scores 0 for every type.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile

CAP_MEM = "4G"
_FENCE = re.compile(r"```([A-Za-z0-9_+.-]*)[^\n]*\n(.*?)```", re.S)
_LANG_TAGS = {"python": {"python", "py", "python3"}, "javascript": {"javascript", "js", "mjs", "node"},
              "cpp": {"cpp", "c++", "cc", "cxx", "hpp", "h"}, "rust": {"rust", "rs"}}


class RewardError(ValueError):
    pass


def validate(spec, where):
    if not isinstance(spec, dict) or spec.get("type") not in ("tests", "exact", "regex", "command"):
        raise RewardError(f"{where}: 'reward' must be an object whose type is tests, exact, regex or command")
    need = {"tests": ("answer_file", "check"), "exact": ("answer",), "regex": ("pattern",), "command": ("argv",)}
    for k in need[spec["type"]]:
        if k not in spec:
            raise RewardError(f"{where}: a {spec['type']} reward needs '{k}'")
    if spec["type"] == "tests":
        if not (isinstance(spec["check"], list) and spec["check"] and all(isinstance(c, list) and c for c in spec["check"])):
            raise RewardError(f"{where}: 'check' must be a list of commands (each a list of strings)")
        if os.path.isabs(spec["answer_file"]) or ".." in spec["answer_file"].split("/"):
            raise RewardError(f"{where}: 'answer_file' must be a plain relative name")
        for name in (spec.get("files") or {}):
            if os.path.isabs(name) or ".." in name.split("/"):
                raise RewardError(f"{where}: file names must be plain relative names")
    if spec["type"] == "regex":
        re.compile(spec["pattern"])


def extract(text, how, language=None):
    text = text or ""
    if how == "all":
        return text.strip()
    if how == "answer_line":
        lines = [l.strip() for l in text.splitlines() if l.strip().startswith("ANSWER:")]
        return lines[-1] if lines else None
    if how == "last_line":
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return lines[-1] if lines else None
    blocks = _FENCE.findall(text)                                   # "fence"
    if not blocks:
        return None
    tags = _LANG_TAGS.get(language or "", set())
    tagged = [b for t, b in blocks if t.lower() in tags]
    return (tagged or [b for _, b in blocks])[-1]


def clean_env():
    """The checks run with none of the caller's PYTHON* settings: Dream starts machx-train with PYTHONSAFEPATH=1 and
    its own PYTHONPATH, and with PYTHONSAFEPATH a test script cannot import the solution.py beside it (every Python
    attempt failed on 2026-11-24 until this)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="random")
    return env


def _capped(cmd):
    return ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={CAP_MEM}", "-p", "MemorySwapMax=0", "--"] + cmd


def _tests(spec, answer):
    d = tempfile.mkdtemp(prefix="grpo_chk_")
    try:
        for name, content in (spec.get("files") or {}).items():
            p = os.path.join(d, name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write(content)
        with open(os.path.join(d, spec["answer_file"]), "w") as f:
            f.write(answer)
        env = clean_env()
        for cmd in spec["check"]:
            try:
                r = subprocess.run(_capped(cmd), cwd=d, capture_output=True, text=True,
                                   timeout=spec.get("timeout", 60), env=env)
            except subprocess.TimeoutExpired:
                return 0.0, "timeout"
            if r.returncode != 0:
                return 0.0, f"check failed (exit {r.returncode})"
        return 1.0, "passed"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def score(task, reply):
    """task: the task line (dict); reply: {"content", "reasoning", "finish_reason"}. -> (reward, short reason)."""
    spec = task["reward"]
    if reply.get("finish_reason") == "length":
        return 0.0, "cut off at the token limit"
    content = reply.get("content") or ""
    t = spec["type"]
    if t == "command":
        try:
            r = subprocess.run(_capped(list(spec["argv"])), input=json.dumps({"task": task, "reply": reply}),
                               capture_output=True, text=True, timeout=spec.get("timeout", 60), env=clean_env())
        except subprocess.TimeoutExpired:
            return 0.0, "reward command timed out"
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        try:
            v = float(lines[-1])
        except (IndexError, ValueError):
            return 0.0, "reward command printed no number"
        return min(1.0, max(0.0, v)), "command"
    default = {"tests": "fence", "exact": "answer_line", "regex": "all"}[t]
    ans = extract(content, spec.get("extract", default), spec.get("language"))
    if ans is None:
        return 0.0, "no answer found in the reply"
    if t == "exact":
        return (1.0, "exact") if ans.strip() == str(spec["answer"]).strip() else (0.0, "wrong answer")
    if t == "regex":
        return (1.0, "matched") if re.search(spec["pattern"], ans) else (0.0, "no match")
    if spec.get("extract", default) == "answer_line":
        ans = ans + "\n"
    return _tests(spec, ans)
