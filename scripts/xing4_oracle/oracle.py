#!/usr/bin/env python3
"""Xing4.0-29B-A4B reference oracle: the released modeling code, fp32 on the CPU.

  python oracle.py --model <HF dir> --out <dir> [--prompt-file f.txt | --ids-file ids.txt] [--chat] [--no-think]
                   [--dtype fp32|bf16] [--streams] [--threads 16]

Writes <out>/ids.i32, logits.f32 ([T, vocab]), routes.i32 ([moe layers, T, top_k]), meta.json and, with --streams,
streams.f32 ([layers + 1, T, hc, dim]: the residual streams entering each layer, then the final ones). The engine's
numerics gate compares against these (KL of the next-token distribution per position, the top token, the routes).
Run with a venv that has transformers >= 5.14 (the model's own code needs it).
"""
import argparse, json, os, sys, time
import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-file", default=""); ap.add_argument("--ids-file", default="")
    ap.add_argument("--chat", action="store_true"); ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"]); ap.add_argument("--streams", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if a.ids_file:
        ids = [int(x) for x in open(a.ids_file).read().split()]
    else:
        text = open(a.prompt_file).read()
        if a.chat:
            msgs = [{"role": "user", "content": text}]
            kw = {"enable_thinking": False} if a.no_think else {}
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            open(os.path.join(a.out, "render.txt"), "w").write(text)
        ids = tok(text, add_special_tokens=False)["input_ids"]
    t0 = time.time()
    dt = torch.float32 if a.dtype == "fp32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=dt, device_map="cpu",
                                                 attn_implementation="eager")
    model.eval()
    cfg = model.config
    print(f"loaded in {time.time() - t0:.0f} s; {len(ids)} tokens; rope_interleave={getattr(cfg, 'rope_interleave', None)}", flush=True)
    routes = []; streams = []
    hooks = []
    for L, layer in enumerate(model.model.layers[: cfg.num_hidden_layers]):
        if hasattr(layer.mlp, "gate"):
            hooks.append(layer.mlp.gate.register_forward_hook(lambda m, i, o: routes.append(o[2].sort(-1).values.to(torch.int32).numpy())))
        if a.streams:
            hooks.append(layer.register_forward_pre_hook(lambda m, args, kw: streams.append(
                (args[0] if args else kw["hidden_states"])[0].float().numpy().copy()), with_kwargs=True))
    if a.streams:
        hooks.append(model.model.layers[cfg.num_hidden_layers - 1].register_forward_hook(
            lambda m, i, o: streams.append((o[0] if isinstance(o, tuple) else o)[0].float().numpy().copy())))
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=False)
    print(f"forward {time.time() - t0:.0f} s", flush=True)
    logits = out.logits[0].float().numpy()
    np.asarray(ids, np.int32).tofile(os.path.join(a.out, "ids.i32"))
    logits.astype(np.float32).tofile(os.path.join(a.out, "logits.f32"))
    if routes: np.stack(routes).tofile(os.path.join(a.out, "routes.i32"))
    if streams: np.stack(streams).astype(np.float32).tofile(os.path.join(a.out, "streams.f32"))
    top = logits.argmax(-1)
    json.dump({"T": len(ids), "vocab": int(logits.shape[1]), "dtype": a.dtype, "moe_layers": len(routes),
               "top_k": int(cfg.num_experts_per_tok), "hc": int(cfg.hc_mult), "dim": int(cfg.hidden_size),
               "layers": int(cfg.num_hidden_layers), "streams": bool(streams),
               "last_top": tok.decode([int(top[-1])])}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("next token after the prompt:", repr(tok.decode([int(top[-1])])), flush=True)

if __name__ == "__main__":
    main()
