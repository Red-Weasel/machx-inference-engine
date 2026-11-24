#!/usr/bin/env python3
"""Rewrite chat-session training files into the shape of Dream's own recorded sessions, without printing any content.

  python3 -I to_dream_format.py (--ref dream_local_train.jsonl | --page-file sample.json) IN.jsonl [IN2.jsonl ...]     -> IN.dream.jsonl beside each

What changes in every record (nothing else; the originals are not touched):
  - the system message becomes Dream's system page: the most common page of --ref (or --page-hash H, the first 8 hex
    digits of the md5 of the page wanted); a record with no system message gets one;
  - `weight` stays only on assistant messages (Dream's records carry it nowhere else); an assistant message with no
    weight gets 1;
  - in run_bash / bash calls every parameter that Dream's run_bash does not take is removed (--keep-params, default
    command,timeout,cwd): e.g. a `description` parameter;
  - `tokens` is re-estimated (characters / 3.5) and `converted_from` records the source file.
Prints counts only.
"""
import argparse, collections, hashlib, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="")
ap.add_argument("--page-hash", default="")
ap.add_argument("--page-file", default="", help="Dream's current page: a text file, or the JSON written by `python -m dream.request_sample` (its \"system\" field)")
ap.add_argument("--keep-params", default="command,timeout,cwd")
ap.add_argument("files", nargs="+")
a = ap.parse_args()
keep = set(a.keep_params.split(","))

if a.page_file:
    page = open(a.page_file).read()
    if a.page_file.endswith(".json"):
        page = json.loads(page)["system"]
    print(f"Dream page from {os.path.basename(a.page_file)}: {len(page)} characters, md5 {hashlib.md5(page.encode()).hexdigest()[:8]}")
else:
    pages = collections.Counter(); text = {}
    for line in open(a.ref):
        m = json.loads(line)["messages"][0]
        if m["role"] == "system":
            h = hashlib.md5(m["content"].encode()).hexdigest()[:8]; pages[h] += 1; text[h] = m["content"]
    h = a.page_hash or pages.most_common(1)[0][0]
    page = text[h]
    print(f"Dream page {h}: {len(page)} characters, used by {pages[h]} of {sum(pages.values())} reference sessions")

CALL = re.compile(r"(<function=(?:run_bash|bash)>)(.*?)(</function>)", re.S)
PARAM = re.compile(r"\s*<parameter=([A-Za-z_]+)>.*?</parameter>", re.S)
dropped = collections.Counter()

def fix_call(m):
    def one(p):
        if p.group(1) in keep:
            return p.group(0)
        dropped[p.group(1)] += 1
        return ""
    return m.group(1) + PARAM.sub(one, m.group(2)) + m.group(3)

for fn in a.files:
    out = re.sub(r"\.jsonl$", "", fn) + ".dream.jsonl"
    n = sys_new = calls = 0
    with open(out, "w") as f:
        for line in open(fn):
            r = json.loads(line); ms = r["messages"]
            if ms and ms[0]["role"] == "system":
                ms[0] = {"role": "system", "content": page}
            else:
                ms.insert(0, {"role": "system", "content": page}); sys_new += 1
            for m in ms:
                if m["role"] == "assistant":
                    m.setdefault("weight", 1)
                    m["content"], k = CALL.subn(fix_call, m["content"]); calls += k
                else:
                    m.pop("weight", None)
            r["tokens"] = int(sum(len(m["content"]) for m in ms) / 3.5)
            r["converted_from"] = os.path.basename(fn)
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    print(f"{os.path.basename(out)}: {n} records, {sys_new} had no system message, {calls} bash calls rewritten, "
          f"{os.path.getsize(out) / 2**20:.1f} MiB")
print("parameters removed from bash calls:", dict(dropped))
