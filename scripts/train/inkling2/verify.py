"""Checks of a trained folder (one model process at a time).

  verify.py bytes  BASE_DIR OUT_DIR                 no model load: header identical, exactly the patched tensors differ
  verify.py run    MODEL_DIR OUT.pt [STATE.pt]      card forward of held-in token ids: the folder as loaded, or the
                                                    base folder with the adapter of a worker state.pt (training forward)
  verify.py cmp    A.pt B.pt                        logits equal?
  verify.py prompt BASE_URL MODEL_DIR SAMPLE.json   the engine's prompt token count vs this package's render
Prints numbers only.
"""
import json, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from inkling2.compressed_experts import SafetensorsReader
from inkling2 import model as M
from inkling2 import data as D

mode = sys.argv[1]
if mode == "bytes":
    a = SafetensorsReader(os.path.join(sys.argv[2], M.WEIGHTS))
    b = SafetensorsReader(os.path.join(sys.argv[3], M.WEIGHTS))
    names = [k for k in a.header if k != "__metadata__"]
    diff = [k for k in names if not np.array_equal(a.raw(k), b.raw(k))]
    want = {f"model.llm.layers.{i}.attn.{M.TARGETS[t][1]}" for i in range(24) for t in M.TARGETS}
    print(json.dumps({"header_identical": a.header_bytes == b.header_bytes, "tensors": len(names), "differ": len(diff),
                      "differ_only_in_the_96_adapted": set(diff) <= want,
                      "experts_identical": not any(".mlp.experts." in k for k in diff),
                      "symlinks": sum(os.path.islink(os.path.join(sys.argv[3], n)) for n in os.listdir(sys.argv[3]))}))
elif mode == "run":
    DEV = os.environ.get("VERIFY_DEVICE", "xpu:0")
    tok, model, tracker, abl = M.load(sys.argv[2], DEV)
    if len(sys.argv) > 4:
        st = torch.load(sys.argv[4], weights_only=False)
        ad = M.add_lora(model)
        with torch.no_grad():
            for k, w in ad.items():
                w.lora_A.copy_(st["A"][k].to(w.lora_A.device))
                w.lora_B.copy_(st["B"][k].to(w.lora_B.device))
        if os.environ.get("VERIFY_CPU_MERGE"):       # W' built on the CPU, as the export builds it
            n = ch = 0
            cpu = {}
            with torch.no_grad():
                for k, w in ad.items():
                    c = (w.base.weight.cpu().float() + w.scale * (st["B"][k] @ st["A"][k])).to(w.base.weight.dtype)
                    d = w.merged_weight().cpu()
                    n += c.numel(); ch += int((c.view(torch.int16) != d.view(torch.int16)).sum())
                    cpu[k] = c.to(w.base.weight.device)

            def fixed(*_a, **_k):
                for k, w in ad.items():
                    w._w = cpu[k]
            model._ie_materialise = fixed
            print(json.dumps({"merged_elements": n, "card_vs_cpu_rounding_differs": ch, "fraction": ch / n}))
    r = json.loads(open(os.path.join(os.environ.get("IE_DATA_DRIVE", os.path.expanduser("~/data")), "dream-dataset/sft/dream_local_eval.jsonl")).readline())
    tools = D.load_sample(D.newest_sample())["tools"]
    msgs, _ = D.convert(r["messages"], tools, 1)
    ids = [x for s in D.Renderer(tok).segments(msgs, tools)[0] for x in s][:int(os.environ.get("VERIFY_T", 1500))]
    with torch.no_grad():
        h = M.lm_forward(model, torch.tensor([ids], device=DEV))
        logits = model.logit_softcap(model.lm_head(h[0, -256:]).float()).cpu()
    torch.save(logits, sys.argv[3])
    print(json.dumps({"tokens": len(ids), "ablation": bool(abl)}))
elif mode == "cmp":
    x, y = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    print(json.dumps({"bit_identical": torch.equal(x, y), "max_abs_diff": (x - y).abs().max().item(),
                      "same_argmax": (x.argmax(-1) == y.argmax(-1)).float().mean().item()}))
elif mode == "prompt":
    from transformers import AutoTokenizer
    url, model_dir, sp = sys.argv[2:5]
    tok = AutoTokenizer.from_pretrained(model_dir)
    s = D.load_sample(sp)
    tools = s["tools"]
    first = s["request"]["messages"]
    tid = "c2_0"
    cases = {"Dream's first request": first,
             "with a call, its result and thinking": first + [
                 {"role": "assistant", "content": "", "reasoning_content": "Look first.", "tool_calls": [
                     {"id": tid, "type": "function", "function": {"name": "tool_schema", "arguments": json.dumps({"search": "files"})}}]},
                 {"role": "tool", "tool_call_id": tid, "content": "result text"}]}
    for name, msgs in cases.items():
        conv = [dict(m) for m in msgs]
        for m in conv:
            if m.get("tool_calls"):
                m["tool_calls"] = [dict(c, function=dict(c["function"], arguments=json.loads(c["function"]["arguments"]))) for c in m["tool_calls"]]
        mine = len(tok.apply_chat_template(conv, tools=tools, tokenize=True, return_dict=False))
        body = {"model": "default", "messages": msgs, "tools": tools, "max_tokens": 1, "stream": False, "temperature": 0}
        rq = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        u = json.load(urllib.request.urlopen(rq, timeout=300))["usage"]
        print(json.dumps({"case": name, "render_tokens": mine, "engine_prompt_tokens": u["prompt_tokens"],
                          "difference": u["prompt_tokens"] - mine}))
