#!/usr/bin/env python3
"""Coaching quiz: problems in the 27B's weak areas, each with a reference answer and hidden tests.

  quiz.py check [ID ...]            every problem's own sanity: the reference passes, a blank / buggy answer fails,
                                    and for test-writing problems the reference tests pass the reference and catch
                                    every bug variant
  quiz.py grade ID FILE             grade one candidate file (the file the task asks the model to write)
  quiz.py list [--jsonl]            the problems (id, area, lang, kind) or their tasks as JSON lines
  quiz.py kit DEST [ID ...]         a folder per problem for a real Dream session: TASK.md + starting files only
                                    (no hidden tests, no reference answers), plus PROMPTS.md
  quiz.py grade-kit DEST            grade the files left in each problem folder; writes DEST/RESULTS.json

Layout, one folder per problem under quiz/:
  meta.json   {"id", "area", "lang": "python"|"js", "kind": "implement"|"fix"|"tests", "file", "timeout"}
  task.md     what the model is shown (fix problems: the broken code is inside it, also in buggy.<ext>)
  solution.*  the reference answer (for "tests": the correct implementation under test)
  test_hidden.*        implement / fix: the grader's tests, import the candidate as `solution`
  reference_tests.*    tests problems: the reference answer (a test file that catches every mutant)
  mutants/*   tests problems: buggy implementations the candidate's tests must catch
A candidate is graded in a fresh temporary folder: `solution.<ext>` + the test file, run with python3 / node.
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(os.environ.get("QUIZ_ROOT") or Path(__file__).resolve().parent / "quiz")
EXT = {"python": ".py", "js": ".mjs", "rust": ".rs", "cpp": ".hpp"}
TEST_EXT = {"python": ".py", "js": ".mjs", "rust": ".rs", "cpp": ".cpp"}
RUN = {"python": ["python3", "-B"], "js": ["node"]}
# compiled languages: the test file is built (with the candidate beside it), then run with the nonce as its argument;
# their tests print argv[1] as their last act. C++ runs under AddressSanitizer + UBSan (leaks and UB fail the test).
BUILD = {"rust": ["rustc", "--edition", "2021", "-O", "-o", "t.bin"],
         "cpp": ["g++", "-std=c++20", "-O1", "-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=undefined",
                 "-pthread", "-o", "t.bin"]}
HOME = os.path.expanduser("~")


def load(pid):
    d = ROOT / pid
    m = json.loads((d / "meta.json").read_text())
    m["dir"] = d
    m["task"] = (d / "task.md").read_text()
    return m


def problems():
    return sorted(p.name for p in ROOT.iterdir() if (p / "meta.json").exists())


# A test file passes only if it runs to its end: the wrapper prints a fresh nonce after the test file returns, so a
# candidate that exits early (sys.exit / process.exit during import) cannot be taken for a pass.
WRAP = {
    "python": ("_quiz_run.py", "import runpy, sys\nrunpy.run_path(sys.argv[1], run_name='__main__')\nprint(sys.argv[2])\n"),
    "js": ("_quiz_run.mjs", "import { pathToFileURL } from 'node:url';\nawait import(pathToFileURL(process.argv[2]).href);\n"
                            "console.log(process.argv[3]);\n"),
}


def _run(m, files, test_name):
    """files: {name: source text}; runs test_name; returns (status, output tail), status 'pass' | 'fail' | 'timeout'."""
    nonce = "QUIZ-DONE-" + os.urandom(8).hex()
    with tempfile.TemporaryDirectory(prefix="quiz-") as t:
        for name, src in files.items():
            Path(t, name).write_text(src)
        env = {"PATH": f"{HOME}/.cargo/bin:" + os.environ.get("PATH", "/usr/bin:/bin"), "HOME": t, "LANG": "C.UTF-8",
               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",   # clean: no PYTHONOPTIMIZE, nothing of the grader's
               "RUSTUP_HOME": os.environ.get("RUSTUP_HOME", f"{HOME}/.rustup"),
               "CARGO_HOME": os.environ.get("CARGO_HOME", f"{HOME}/.cargo"), "ASAN_OPTIONS": "detect_leaks=1"}
        if m["lang"] in BUILD:
            try:
                b = subprocess.run(BUILD[m["lang"]] + [test_name], cwd=t, capture_output=True, timeout=180, env=env)
            except subprocess.TimeoutExpired:
                return "fail", "compile timed out"
            except FileNotFoundError as e:
                return "fail", f"compiler not found: {e}"
            if b.returncode != 0:
                return "fail", "compile failed:\n" + (b.stdout + b.stderr).decode("utf-8", "replace")[-1500:]
            cmd = ["./t.bin", nonce]
        else:
            wname, wsrc = WRAP[m["lang"]]
            Path(t, wname).write_text(wsrc)
            cmd = RUN[m["lang"]] + [wname, test_name, nonce]
        try:
            r = subprocess.run(cmd, cwd=t, capture_output=True, timeout=m.get("timeout", 20), env=env)
        except subprocess.TimeoutExpired:
            return "timeout", f"timed out after {m.get('timeout', 20)} s"
        out = (r.stdout + r.stderr).decode("utf-8", "replace")
        done = r.returncode == 0 and nonce in r.stdout.decode("utf-8", "replace")
        return ("pass" if done else "fail"), out.replace(nonce, "").strip()[-1500:]


def grade_source(m, src):
    """Grade candidate source text. Returns {"pass": bool, "detail": str, ...}."""
    ext = EXT[m["lang"]]
    if m["kind"] == "tests":
        ref = (m["dir"] / f"solution{ext}").read_text()
        st, out = _run(m, {f"solution{ext}": ref, m["file"]: src}, m["file"])
        if st != "pass":
            return {"pass": False, "detail": "your tests fail on a CORRECT implementation:\n" + out, "caught": 0}
        missed = []
        muts = sorted((m["dir"] / "mutants").iterdir())
        for mu in muts:
            st, _ = _run(m, {f"solution{ext}": mu.read_text(), m["file"]: src}, m["file"])
            if st != "fail":            # a pass misses the bug; a timeout proves nothing either
                missed.append(mu.stem)
        return {"pass": not missed, "caught": len(muts) - len(missed), "of": len(muts),
                "detail": "caught every buggy version" if not missed else "missed buggy versions: " + ", ".join(missed)}
    test = f"test_hidden{TEST_EXT[m['lang']]}"
    st, out = _run(m, {f"solution{ext}": src, test: (m["dir"] / test).read_text()}, test)
    return {"pass": st == "pass", "detail": "all hidden tests pass" if st == "pass" else out}


def check(ids):
    bad = 0
    for pid in ids:
        m = load(pid); ext = EXT[m["lang"]]; notes = []
        if m["kind"] == "tests":
            r = grade_source(m, (m["dir"] / f"reference_tests{ext}").read_text())
            good = r["pass"]; notes.append(f"reference tests catch {r.get('caught')}/{r.get('of')} bug variants"
                                           + ("" if good else f" ({r['detail'][:300]})"))
            blank = grade_source(m, "")
            if blank["pass"]: good = False; notes.append("an EMPTY test file passes (mutants not caught)")
        else:
            r = grade_source(m, (m["dir"] / f"solution{ext}").read_text())
            good = r["pass"]; notes.append("reference passes" if good else "REFERENCE FAILS: " + r["detail"][-400:])
            for neg in (["buggy" + ext] if m["kind"] == "fix" else []) + ["<blank>"]:
                src = "" if neg == "<blank>" else (m["dir"] / neg).read_text()
                if grade_source(m, src)["pass"]: good = False; notes.append(f"{neg} PASSES the hidden tests (tests too weak)")
                else: notes.append(f"{neg} fails")
        bad += not good
        print(f"{'ok  ' if good else 'BAD '} {pid:28s} {m['area']:12s} {m['kind']:9s} | " + "; ".join(notes), flush=True)
    print(f"{len(ids) - bad}/{len(ids)} problems sound")
    return bad == 0


RUN_HINT = {"python": "python3 {f}", "js": "node {f}",
            "rust": "a small main.rs with `mod solution;` built with `rustc --edition 2021 main.rs`",
            "cpp": "a small test.cpp that includes solution.hpp, built with `g++ -std=c++20 test.cpp`"}


def kit(dest, ids):
    dest = Path(dest).expanduser(); dest.mkdir(parents=True, exist_ok=True)
    lines = ["# Quiz prompts for Dream", "",
             "One NEW conversation per problem, with that problem's folder as the workspace. Paste the prompt as",
             "the first message. Let it work until it says it",
             "is done. Then grade everything with:", "",
             f"    python3 \"{Path(__file__).resolve()}\" grade-kit \"{dest}\"", ""]
    for pid in ids:
        m = load(pid); ext = EXT[m["lang"]]; d = dest / pid; d.mkdir(exist_ok=True)
        if m["kind"] == "fix":
            (d / m["file"]).write_text((m["dir"] / f"buggy{ext}").read_text())
        elif m["kind"] == "tests":
            (d / f"solution{ext}").write_text((m["dir"] / f"solution{ext}").read_text())
        (d / "TASK.md").write_text(m["task"])
        # the session's workspace IS the problem folder (Dream runs each problem inside its own folder): name no subfolder
        prompt = (f"Read TASK.md in this folder and do what it asks. Write `{m['file']}` here"
                  + (" (the current version is already here)" if m["kind"] == "fix" else "")
                  + f". Run it to check your work (for example {RUN_HINT[m['lang']].format(f=m['file'])}), "
                  "fix anything that fails, and tell me when it is done.")
        lines += [f"## {pid}  ({m['area']}, {m['lang']}, {m['kind']})", "", prompt, ""]
    (dest / "PROMPTS.md").write_text("\n".join(lines))
    print(f"kit: {len(ids)} problems in {dest}; prompts in {dest / 'PROMPTS.md'}")


def tests_revision():
    """A short hash of everything that decides a grade: each problem's task, meta, hidden tests, reference tests and
    mutants (not the reference solutions)."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(ROOT.rglob("*")):
        if not f.is_file() or f.name.startswith("solution.") or "__pycache__" in f.parts: continue
        h.update(str(f.relative_to(ROOT)).encode()); h.update(b"\0"); h.update(f.read_bytes()); h.update(b"\0")
    return h.hexdigest()[:10]


def grade_kit(dest):
    dest = Path(dest).expanduser(); res = []
    for d in sorted(x for x in dest.iterdir() if x.is_dir() and (ROOT / x.name / "meta.json").exists()):
        m = load(d.name); f = d / m["file"]
        if m["kind"] == "fix" and f.is_file() and f.read_text() == (m["dir"] / f"buggy{EXT[m['lang']]}").read_text():
            r = {"pass": False, "detail": "file unchanged (not attempted)"}
        else:
            r = grade_source(m, f.read_text(errors="replace")) if f.is_file() else {"pass": False, "detail": f"{m['file']} not written"}
        res.append({"id": d.name, "area": m["area"], "lang": m["lang"], "kind": m["kind"], **r})
        print(f"{'PASS' if r['pass'] else 'fail'} {d.name:26s} {m['area']:15s} | {r['detail'][:100].replace(chr(10), ' ')}")
    print(f"{sum(r['pass'] for r in res)}/{len(res)} passed")
    # 2026-11-21 (the Codex review of a grader change made during a run): every grading records WHICH tests graded it, and
    # a grading under other tests never overwrites the earlier one. RESULTS.json stays the list Dream reads; beside it,
    # RESULTS.meta.json = {"tests": <revision of every hidden test and task>, "passed", "graded"}, and when the kit was
    # graded before under another revision that grading is kept as RESULTS.<its revision>.json (+ its meta).
    rev = tests_revision()
    out, meta = dest / "RESULTS.json", dest / "RESULTS.meta.json"
    old_rev = None
    if out.exists():
        try: old_rev = json.loads(meta.read_text())["tests"] if meta.exists() else "unrecorded"
        except Exception: old_rev = "unrecorded"
        if old_rev != rev:
            keep = dest / f"RESULTS.{old_rev}.json"
            if not keep.exists():
                keep.write_text(out.read_text())
                if meta.exists(): (dest / f"RESULTS.{old_rev}.meta.json").write_text(meta.read_text())
            print(f"note: this kit was graded before under tests {old_rev}; that grading is kept as {keep.name}")
    out.write_text(json.dumps(res, indent=1))
    import time as _t
    meta.write_text(json.dumps({"tests": rev, "passed": sum(r["pass"] for r in res), "of": len(res),
                                "graded": _t.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    print(f"tests revision {rev}")


def main():
    a = sys.argv[1:]
    if not a or a[0] == "check":
        sys.exit(0 if check(a[1:] or problems()) else 1)
    if a[0] == "grade":
        m = load(a[1]); r = grade_source(m, Path(a[2]).read_text())
        print(json.dumps(r, indent=1)); sys.exit(0 if r["pass"] else 1)
    if a[0] == "kit":
        kit(a[1], a[2:] or problems()); return
    if a[0] == "grade-kit":
        grade_kit(a[1]); return
    if a[0] == "list":
        for pid in problems():
            m = load(pid)
            if "--jsonl" in a:
                print(json.dumps({k: m[k] for k in ("id", "area", "lang", "kind", "file", "task")}))
            else:
                print(f"{pid:28s} {m['area']:12s} {m['lang']:7s} {m['kind']}")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "revision":
        print(tests_revision()); sys.exit(0)
    print(__doc__); sys.exit(2)


if __name__ == "__main__":
    main()
