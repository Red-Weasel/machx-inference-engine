import json, os, sys
sys.path.insert(0, "scripts/dream_bench")
import toolcall_bench as tb
X = os.environ.get("XG4_WORK", "/tmp/xing4-heldout")
op = json.load(open(X + "/dream_opening.json"))
res = json.load(open(X + "/bench_q8_guard3.json"))["results"]
seen, turns = set(), []
for r in res:
    if r["correct"] and r["prompt"] not in seen and len(turns) < 18:
        seen.add(r["prompt"]); turns.append(r)
msgs = [{"role": "system", "content": op["system"]}]
for i, r in enumerate(turns):
    msgs.append({"role": "user", "content": r["prompt"]})
    msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": r["call"]["function"]}]})
    msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
msgs.append({"role": "assistant", "content": "All done."})
json.dump({"messages": msgs, "tools": tb.TOOLS}, open(X + "/heldout.jsonl", "w")); open(X + "/heldout.jsonl", "a").write("\n")
print(len(turns), "turns")
