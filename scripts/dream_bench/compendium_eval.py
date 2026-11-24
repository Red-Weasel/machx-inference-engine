#!/usr/bin/env python3
"""Coding score on the owner's coding compendium: each exercise's function, written by the model, graded by the
compendium's own self-checking solution program (brute-force / reference checks), run by its own tools/verify.py.

  compendium_eval.py --check                       # grader sanity: every reference function spliced back must pass
  compendium_eval.py --url http://127.0.0.1:11499 --label 27B --out r.json [--limit N] [--parallel 4] [--temp 0.2]

How a task is made (any language the compendium runs): the exercise's statement (text before its solution) names the
function to write (the first backticked `name(`...); the reference solution program defines it. The model is asked for
that definition only (plus helpers); the grader replaces the reference's definition with the model's and runs the program
(C/C++ under ASan+UBSan, Rust with overflow checks, as verify.py does). Pass = the program's own checks pass.
Exercises with no named function, or whose reference does not define it, are skipped (counted).
"""
import argparse, concurrent.futures as cf, glob, json, os, re, sys, time, urllib.request
from pathlib import Path

C_LIKE = {"c", "cpp", "rust", "js", "ts", "java", "go"}


def load_verify(root):
    sys.path.insert(0, str(Path(root) / "tools"))
    import verify  # the compendium's runner
    return verify


# ------------------------------------------------------------------------------------------------ definition spans
def def_span(code, name, lang):
    """(start, end) character span of the top-level definition of `name` in `code`, or None."""
    if lang == "python":
        m = re.search(rf"^(?P<ind>[ \t]*)(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b", code, re.M)
        if not m: return None
        ind = len(m.group("ind")); start = m.start()
        # decorators directly above
        while True:
            prev = code.rfind("\n", 0, start - 1)
            line = code[prev + 1:start - 1] if start > 0 else ""
            if line.strip().startswith("@") and len(line) - len(line.lstrip()) == ind: start = prev + 1
            else: break
        pos = code.find("\n", m.end())
        if pos < 0: return start, len(code)
        end = len(code)
        for lm in re.finditer(r"^.*$", code[pos + 1:], re.M):
            ln = lm.group(0)
            if ln.strip() and (len(ln) - len(ln.lstrip())) <= ind and not ln.lstrip().startswith(("#", ")", "]")):
                end = pos + 1 + lm.start(); break
        return start, end
    # C-like: name ( ... ) [qualifiers / -> ret] {   -- the opening brace of the body, then brace matching
    for m in re.finditer(rf"\b{re.escape(name)}\s*(<[^>{{;]*>)?\s*\(", code):
        # not a call: the previous significant text on the line is a type / keyword, and a '{' follows the ')'
        depth, i = 0, m.end() - 1
        while i < len(code):
            if code[i] == "(": depth += 1
            elif code[i] == ")":
                depth -= 1
                if depth == 0: break
            i += 1
        j = i + 1
        while j < len(code) and code[j] not in "{;=": j += 1
        if j >= len(code) or code[j] != "{": continue
        line_start = code.rfind("\n", 0, m.start()) + 1
        head = code[line_start:m.start()]
        if re.search(r"[=(,]\s*$|\breturn\b|\.\s*$", head): continue   # a call or an expression
        # template<...> / attributes / doc comments on the lines directly above stay with the definition
        start = line_start
        while True:
            prev = code.rfind("\n", 0, start - 1) + 1 if start > 0 else 0
            pl = code[prev:start].strip()
            if start > 0 and (pl.startswith("template") or pl.startswith("#[") or pl.startswith("[[") or pl.startswith("@")): start = prev
            else: break
        depth, k = 0, j
        while k < len(code):
            if code[k] == "{": depth += 1
            elif code[k] == "}":
                depth -= 1
                if depth == 0: break
            k += 1
        end = code.find("\n", k)
        return start, (len(code) if end < 0 else end + 1)
    return None


def target_name(statement):
    for m in re.finditer(r"`([^`]+)`", statement):
        s = m.group(1)
        mm = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(<[^>]*>)?\s*\(", s)
        if mm and mm.group(1) not in {"if", "for", "while", "return", "sizeof", "assert", "print", "printf", "main"}:
            return mm.group(1)
    return None


# ------------------------------------------------------------------------------------------------ tasks
def build_tasks(root, verify):
    tasks, skipped = [], {"no_name": 0, "not_defined": 0}
    for f in sorted(glob.glob(str(Path(root) / "chapters" / "*" / "*exercise*.md"))):
        text = Path(f).read_text()
        lines = text.split("\n")
        blocks, _, _ = verify.parse_markdown(Path(f))
        for b in blocks:
            if b.lang not in verify.LANGS: continue
            lang = verify.LANGS[b.lang][0]
            # the exercise: from the nearest heading above the block to the solution marker / the block
            i = b.line - 2
            while i >= 0 and not re.match(r"#{2,4} ", lines[i]): i -= 1
            if i < 0: continue
            sec = "\n".join(lines[i:b.line - 1])
            title = lines[i].lstrip("# ").strip()
            stmt = re.split(r"\n\*\*Solution\b|\n#{3,5} Solution|\nSolution[.:]", sec)[0]
            stmt = re.sub(r"\n```.*", "", stmt, flags=re.S).strip()
            name = target_name(stmt)
            if not name: skipped["no_name"] += 1; continue
            sp = def_span(b.code, name, lang)
            if not sp: skipped["not_defined"] += 1; continue
            if any(t["file"] == f and t["name"] == name for t in tasks): continue
            tasks.append({"id": f"{Path(f).stem}:{b.line}", "file": f, "line": b.line, "title": title, "lang": lang, "info": b.info,
                          "name": name, "statement": stmt, "reference": b.code, "span": list(sp), "attrs": b.attrs,
                          "expects": b.expects, "flags": sorted(b.flags_present)})
    return tasks, skipped


def run_program(verify, task, code):
    b = verify.Block(path=Path(task["file"]), line=task["line"], lang=task["lang"] if task["lang"] in verify.LANGS else "python",
                     info=task["info"], code=code, attrs=dict(task["attrs"]), expects=list(task["expects"]),
                     flags_present=set(task["flags"]))
    # Block.lang holds the info alias; the canonical names are keys of LANGS too
    r = verify.build_and_run(b)
    return r


IMPORT_LINE = {   # top-of-file lines a model writes with its definitions; spliced mid-file they break the build
    "java": re.compile(r"^\s*import\s+[\w.*]+\s*;\s*$"),
    "c": re.compile(r"^\s*#\s*include\b.*$"), "cpp": re.compile(r"^\s*#\s*include\b.*$"),
    "rust": re.compile(r"^use\s+[^;]+;\s*$"),
    "js": re.compile(r"^import\s.+?from\s+['\"][^'\"]+['\"]\s*;?\s*$|^import\s+['\"][^'\"]+['\"]\s*;?\s*$"),
    "ts": re.compile(r"^import\s.+?from\s+['\"][^'\"]+['\"]\s*;?\s*$"),
    "python": re.compile(r"^from\s+__future__\s+import\s.+$"),   # must be the file's first statement
}


def _hoist(task, ref_before, mc):
    """Move the model's import / #include / use lines out of mc; put the ones the file lacks at its top."""
    rx = IMPORT_LINE.get(task["lang"])
    if not rx: return ref_before, mc
    keep, hoisted = [], []
    for ln in mc.split("\n"):
        (hoisted if rx.match(ln) else keep).append(ln)
    if not hoisted: return ref_before, mc
    have = {l.strip() for l in task["reference"].split("\n")}
    new = [h.strip() for h in hoisted if h.strip() not in have]
    if not new: return ref_before, "\n".join(keep)
    head = "\n".join(dict.fromkeys(new)) + "\n"
    if task["lang"] == "java":   # after a package line, if any
        m = re.match(r"(\s*package\s+[\w.]+\s*;\s*\n)", ref_before)
        if m: return m.group(1) + head + ref_before[m.end():], "\n".join(keep)
    return head + ref_before, "\n".join(keep)


def splice_variants(task, model_code):
    """Programs to try, in order: (variant, source). 'whole' = the model's code at the reference definition's place
    (imports hoisted); 'def' = only the model's own definition of the target name (when it wrote more around it, e.g.
    a whole class around a method the test file defines inside its own class)."""
    s, e = task["span"]
    ref = task["reference"]
    mc = model_code.strip("\n") + "\n"
    if task["lang"] == "go":   # imports must stay at the top: drop the model's (the reference's cover the usual ones)
        mc = re.sub(r"^import\s*(\([^)]*\)|\"[^\"]*\")\s*$", "", mc, flags=re.M)
    out = []
    before, body = _hoist(task, ref[:s], mc)
    out.append(("whole", before + body + ref[e:]))
    sp = def_span(body, task["name"], task["lang"])
    if sp and body[sp[0]:sp[1]].strip() != body.strip():
        out.append(("def", before + body[sp[0]:sp[1]].strip("\n") + "\n" + ref[e:]))
    return out


def splice(task, model_code):
    return splice_variants(task, model_code)[0][1]


def grade(verify, task, code):
    """Run each splice variant until one passes; returns (result of the passing or the first variant, variant name)."""
    first = None
    for name, prog in splice_variants(task, code):
        r = run_program(verify, task, prog)
        if r.status == "PASS": return r, name
        if first is None: first = (r, name)
    return first


FENCE = re.compile(r"```[A-Za-z0-9+#]*[^\n]*\n(.*?)```", re.S)


def ask(url, task, temp, think, max_tokens):
    lang_name = {"cpp": "C++", "js": "JavaScript", "ts": "TypeScript", "c": "C", "go": "Go", "rust": "Rust",
                 "python": "Python", "java": "Java", "bash": "Bash", "sql": "SQL"}.get(task["lang"], task["lang"])
    prompt = (f"{task['statement']}\n\nWrite the {lang_name} definition of `{task['name']}` (and any helper functions or types it "
              f"needs). Reply with one {lang_name} code block containing only those definitions: no main function, no tests, "
              f"no example usage.")
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "temperature": temp, "top_p": 0.95,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": think}}
    t0 = time.time()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    msg = r["choices"][0]["message"]; text = msg.get("content") or ""
    blocks = FENCE.findall(text)
    code = max(blocks, key=len) if blocks else text
    return code, text, time.time() - t0, r.get("usage", {}).get("completion_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compendium", default=os.path.expanduser("~/Desktop/00 - data training/coding-compendium"))
    ap.add_argument("--check", action="store_true", help="grader sanity: splice each reference definition back in")
    ap.add_argument("--url", default="http://127.0.0.1:11499")
    ap.add_argument("--label", default="model"); ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.2); ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--regrade", default="", help="re-grade the answers saved in an earlier --out file (no model calls)")
    a = ap.parse_args()
    verify = load_verify(a.compendium)
    tasks, skipped = build_tasks(a.compendium, verify)
    saved = {}
    if a.regrade:
        saved = {r["id"]: r for r in json.load(open(a.regrade))["results"]}
        tasks = [t for t in tasks if t["id"] in saved]
    if a.limit: tasks = tasks[:a.limit]
    langs = {}
    for t in tasks: langs[t["lang"]] = langs.get(t["lang"], 0) + 1
    print(f"{len(tasks)} gradable exercises {langs}; skipped {skipped}", flush=True)
    results = []

    def one(t):
        if a.check:
            s, e = t["span"]; code = t["reference"][s:e]; text = ""; secs = 0; ntok = 0
        elif a.regrade:
            o = saved[t["id"]]; text = o.get("answer", ""); secs = float(o.get("secs", 0) or 0); ntok = int(o.get("tokens", 0) or 0)
            if len(text) >= 4000 and text.count("```") % 2:   # saved answer was cut at 4000 chars: keep its grade
                return dict(o, pass_=None, variant="kept (answer cut in the saved file)") | {"pass": o["pass"] in (True, "True")}
            blocks = FENCE.findall(text); code = max(blocks, key=len) if blocks else text
        else:
            try: code, text, secs, ntok = ask(a.url, t, a.temp, a.think, a.max_tokens)
            except Exception as x: return {"id": t["id"], "lang": t["lang"], "pass": False, "why": f"request: {x}"[:200]}
        r, variant = grade(verify, t, code)
        ok = r.status == "PASS"
        return {"id": t["id"], "title": t["title"], "lang": t["lang"], "name": t["name"], "pass": ok, "status": r.status,
                "variant": variant,
                "why": "" if ok else (r.status + ": " + (r.detail or "") + " " + (r.output or "")[-200:])[:400], "secs": round(secs, 1), "tokens": ntok,
                "answer": text[:40000]}

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=1 if a.check else a.parallel) as ex:
        for res in ex.map(one, tasks): results.append(res)
    n = len(results); p = sum(r["pass"] for r in results)
    per = {}
    for r in results:
        d = per.setdefault(r["lang"], [0, 0]); d[0] += r["pass"]; d[1] += 1
    head = "grader check (reference spliced back)" if a.check else a.label
    print(f"{head}: {p}/{n} = {100.0 * p / max(n, 1):.1f} % in {time.time() - t0:.0f} s | " +
          ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(per.items())))
    for r in results:
        if not r["pass"]: print("  FAIL", r["id"], r.get("name", ""), "|", r["why"][:160].replace("\n", " "))
    if a.out: json.dump({"label": head, "passed": p, "n": n, "per_lang": per, "skipped": skipped, "results": results}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
