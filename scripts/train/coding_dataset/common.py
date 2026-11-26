"""Shared parts of the coding dataset generator: task loading, the sandbox around a Dream run, and grading.

A run happens inside bubblewrap: no network (the engine is reached through one forwarded socket), an empty home, no view
of the task folders, so the hidden tests cannot be seen. Every run and every test command has its own memory-capped scope.
"""
import os
import json, os, shutil, subprocess, sys, uuid

WORK = os.path.join(os.environ.get("IE_DATA_DRIVE", os.path.join("/media", os.environ.get("USER", ""), "Secondary Drive")), "dream-coding-work")
RECORDS = os.path.join(WORK, "records")
CHECKED = os.path.expanduser("~/models/inkling2-train/data/checked")
DREAM = os.environ.get("DREAM_DIR", os.path.expanduser("~/Desktop/Dream-Agent-Harness"))
NODE = os.environ.get("DREAM_NODE", os.path.expanduser("~/.nvm/versions/node/v24.15.0"))
SLOTS = "/tmp/dream-coding"
PLAYWRIGHT = os.path.expanduser("~/.cache/ms-playwright")
LEASES = "/tmp/dream-coding/leases"
HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "page.md")
MAX_FILE_KB = 256 * 1024
SAMPLING = ["--temperature", "1.0", "--top-p", "1.0", "--top-k", "0", "--min-p", "0", "--repeat-penalty", "1.0"]
MAX_TOKENS = 32768                               # as the direct test; 10000 cut 7 replies mid-thought


def load_tasks():
    sys.path.insert(0, os.path.join(CHECKED, "build"))
    import harness
    split = {}
    for line in open(os.path.join(CHECKED, "manifest.jsonl")):
        r = json.loads(line)
        split[r["id"]] = r["split"]
    out = {}
    for m in harness.load_family_modules():
        for t in m.tasks():
            t = dict(t, split=split[t["id"]])
            out[t["id"]] = t
    return out


def _base(extra):
    return ["bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin", "--ro-bind", "/etc", "/etc",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/home/user", "--ro-bind", NODE, "/opt/node",
            "--unshare-all", "--die-with-parent", "--new-session", "--clearenv", "--setenv", "HOME", "/home/user",
            "--setenv", "PATH", "/opt/node/bin:/usr/bin:/bin", "--setenv", "LANG", "C.UTF-8", "--setenv", "TMPDIR", "/tmp"] + extra


def scope(cap, cmd):
    return ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={cap}", "-p", "MemorySwapMax=0",
            "-p", "TasksMax=512", "--"] + cmd


def seed_workspace(t, ws):
    """A task that hands the model a file to fix ("The header `solution.hpp` below has bugs" + one code block) gets that
    file in the workspace, as it would be in a real project: without it the model looked for the file, found nothing,
    and wrote the fix into its reply (5 of the 18 tasks Dream lost that a direct answer passed)."""
    import re
    if t["language"] == "text" or t["answer_file"] not in t["statement"]:
        return False
    blocks = re.findall(r"```[a-zA-Z0-9+]*\n(.*?)```", t["statement"], re.S)
    if len(blocks) != 1:
        return False
    p = os.path.join(ws, t["answer_file"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(blocks[0])
    return True


COACH_MARK = "Coach's note for this task:"


def strip_coach(msgs):
    """The session as if no coach had spoken: the note is cut from the task prompt (training learns the habit itself)."""
    out = []
    for m in msgs:
        c = m.get("content")
        if m.get("role") == "user" and isinstance(c, str) and COACH_MARK in c:
            m = dict(m, content=c[:c.index(COACH_MARK)].rstrip() + ("\n" if c.endswith("\n") else ""))
        elif m.get("role") == "user" and isinstance(c, list):
            m = dict(m, content=[dict(p, text=p["text"][:p["text"].index(COACH_MARK)].rstrip())
                                 if isinstance(p, dict) and COACH_MARK in (p.get("text") or "") else p for p in c])
        out.append(m)
    return out


def dream_cmd(slot, minutes, inner_args=None):
    """The Dream run of one worker slot, in a memory scope and a Landlock domain. Workspace, record and home are the
    slot's own folders under SLOTS (/tmp: Dream refuses instruction files under a parent with an ACL, as the data drive has)."""
    w, rec, home, tmp = (os.path.join(SLOTS, f"{x}{slot:02d}") for x in "wrht")
    ro = ["/usr", "/etc", "/proc", "/sys", DREAM, NODE, PLAYWRIGHT]
    rw = [w, rec, home, tmp, LEASES, "/dev/null", "/dev/zero", "/dev/urandom", "/dev/random", "/dev/tty", "/dev/shm", "/dev/pts"]
    env = {"HOME": home, "PATH": f"{NODE}/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "TMPDIR": tmp, "PYTHONDONTWRITEBYTECODE": "1",
           "PLAYWRIGHT_BROWSERS_PATH": PLAYWRIGHT, "DREAM_INFERENCE_LEASE_DIR": os.path.join(LEASES, f"{slot:02d}")}
    # one lease folder per slot: Dream refuses a request while another holds the lease in the same folder (one local
    # request at a time), but this engine serves --parallel 12 conversations; a shared folder refused 37 of 125 runs
    inner = (f"ulimit -f {MAX_FILE_KB}; cd {w} && exec {DREAM}/.venv/bin/python -m " + (inner_args or (
             f"dream.task_run --workspace {w} --record {rec} --root {rec}/root --prompt-file {rec}/prompt.md --name task "
             f"--minutes {minutes} --max-tokens {MAX_TOKENS} " + " ".join(SAMPLING))))
    ll = ["/usr/bin/python3", "-I", os.path.join(HERE, "ll_exec.py")]
    for p in ro:
        ll += ["--ro", p]
    for p in rw:
        ll += ["--rw", p]
    # a session cut off at its time limit leaves its lease "uncertain"; Dream clears that only once the engine has
    # nothing in flight, which never happens while the other slots run, so every later session in the slot was
    # refused (36 of 156 in one held-out check). The slot's previous session is always over here: start it clean.
    shutil.rmtree(os.path.join(LEASES, f"{slot:02d}"), ignore_errors=True)
    os.makedirs(os.path.join(LEASES, f"{slot:02d}"), mode=0o700)
    ll += ["--ro", HERE, "--port", "11435", "--"]
    return scope("8G", ["env", "-i", *[f"{k}={v}" for k, v in env.items()], "PYTHONPATH=" + DREAM, *ll, "bash", "-c", inner]), w, rec, home, tmp


def text_answer(msgs):
    """The model's answer to a one-answer task: its last assistant text that is exactly one ANSWER line."""
    for m in reversed(msgs):
        c = m.get("content") if m.get("role") == "assistant" else None
        if isinstance(c, str):
            c = c.rsplit("</think>", 1)[-1].strip()
            if c.startswith("ANSWER:") and "\n" not in c:
                return c
    return None


def grade(t, final_text, ws):
    """Run the task's own hidden tests on the workspace's answer. -> (passed, reason)"""
    if t["language"] == "text":
        answer = (final_text or "").strip()
    else:
        p = os.path.join(ws, t["answer_file"])
        if os.path.islink(p) or not os.path.isfile(p):
            return False, "no_answer_file"
        if os.path.getsize(p) > 2 << 20:
            return False, "answer_too_large"
        answer = open(p, errors="replace").read()
    tries = 2 if t["family"].endswith("_perf") else 1             # timing tests: once more before calling it a failure
    reason = ""
    for _ in range(tries):
        ok, reason = _check(t, answer)
        if ok:
            return True, ""
    return False, reason


def _check(t, answer):
    d = os.path.join(WORK, "grade", uuid.uuid4().hex)
    os.makedirs(d)
    try:
        for name, content in t["files"].items():
            p = os.path.join(d, name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(content)
        open(os.path.join(d, t["answer_file"]), "w").write(answer)
        for cmd in t["check"]:
            full = scope("4G", _base(["--bind", d, "/t", "--chdir", "/t", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
                                      "--setenv", "ASAN_OPTIONS", "detect_leaks=1", "--setenv", "PYTHONHASHSEED", "random",
                                      "bash", "-c", f'ulimit -f {MAX_FILE_KB}; exec "$@"', "x"] + cmd))
            try:
                r = subprocess.run(full, capture_output=True, text=True, errors="replace", timeout=t.get("timeout", 60))
            except subprocess.TimeoutExpired:
                return False, "test_timeout"
            if r.returncode != 0:
                return False, "test_failed"
        return True, ""
    finally:
        shutil.rmtree(d, ignore_errors=True)
