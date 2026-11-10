#!/usr/bin/env python3
"""Generic mini-model oracle: any architecture that transformers builds and llama.cpp's converter converts.

The qwen4exp oracle (scripts/qwen4exp_oracle/mini_model.py) writes its GGUF by hand. This one uses the real converter, so
it needs no per-arch writer and also tests the converter path users take:

  build    a tiny random model from a transformers config (real widths, few layers), every >= 2-D weight whose last
           dim is a multiple of 32 put on the Q8_0 grid first, everything else on the fp16 grid (so llama.cpp's
           --outtype q8_0 is lossless and the reference sees the engine's exact weights),
           saved with save_pretrained + a real tokenizer, converted by convert_hf_to_gguf.py
  run      ie serve on one card / one lane with IE_LOGITS_DUMP: a chat completion (greedy) dumps the prompt ids, every
           step's logits (step 0 = the prompt's last row) and the picks
  compare  the transformers model in fp32 on the prompt + the engine's picks (teacher-forced): every step's logits
           against the engine's -- cosine >= --min-cos on every row, argmax equal where the reference's top-1 is decisive

  python generic_mini.py build --arch qwen3 --out D [--layers 4] [--set hidden_size=2048 ...] --tokenizer-dir T
  python generic_mini.py requant --out D --type Q4_K_M        (-> D-Q4_K_M; the reference = that GGUF, dequantized)
  python generic_mini.py run --out D [--gpu 0] [--tokens 16]
  python generic_mini.py compare --out D [--negative]

The gate: the prefill row (step 0) cos >= 0.9999 (the weights are exact and the model shallow: qwen3 4 layers measured
1 - cos 4e-6), every decode row >= 0.999 (int8-activation decode kernels sit near 1 - cos 6e-4 by design), argmax misses
<= 1. --negative scales one middle layer's 1-D gammas by 1.1 in the reference and MUST fail (qwen3 measured
1 - cos 5.5e-3, 2 argmax misses) -- run it once per new arch, or the gate proves nothing there.
The logits come from IE_LOGITS_DUMP, which hooks Engine::generate (the one-lane dense path ie serve --parallel 1 takes).

Env: LLAMA_CPP (default ~/llama.cpp) for convert_hf_to_gguf.py + gguf-py; IE_BIN (default build/src/ie of this tree).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LLAMA = os.environ.get("LLAMA_CPP", os.path.expanduser("~/llama.cpp"))


def q8_grid_(t):
    """In place: put a >= 2-D tensor whose last dim is a multiple of 32 on the Q8_0 grid (blocks of 32 along the last dim,
    fp16 scale = amax / 127, q = round half away from zero) -- what llama.cpp's q8_0 quantizer reproduces exactly."""
    import torch
    if t.dim() < 2 or t.shape[-1] % 32:
        return False
    x = t.detach().double().reshape(-1, 32)
    d = (x.abs().amax(dim=1, keepdim=True) / 127.0).half().double()
    q = torch.where(d > 0, x / torch.where(d > 0, d, torch.ones_like(d)), torch.zeros_like(x))
    q = torch.sign(q) * torch.floor(q.abs() + 0.5)
    t.data.copy_((q.clamp(-127, 127) * d).reshape(t.shape).to(t.dtype))
    return True


def parse_sets(sets):
    out = {}
    for s in sets or []:
        k, v = s.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def cmd_build(a):
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    os.makedirs(a.out, exist_ok=True)
    hf = os.path.join(a.out, "hf")
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
    vocab = max(len(tok), getattr(tok, "vocab_size", 0))
    over = dict(num_hidden_layers=a.layers, vocab_size=vocab, tie_word_embeddings=False)
    over.update(parse_sets(a.set))
    cfg = AutoConfig.for_model(a.arch, **over)
    if tok.eos_token_id is not None:
        cfg.eos_token_id = tok.eos_token_id
    torch.manual_seed(a.seed)
    m = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    gen = torch.Generator().manual_seed(a.seed + 1)
    n_grid = n_norm = 0
    with torch.no_grad():
        n_router = 0
        for name, p in m.named_parameters():
            # MoE routers scaled up: the softmax turns peaked, so an expert swapped at the k-th place (random routers leave the
            # k-th / (k+1)-th logits close, and the engine's fp16 input can tip them) carries little weight. It does NOT make
            # such a swap rarer -- the gap and the rounding error scale together.
            if a.router_scale != 1.0 and p.dim() == 2 and (name.endswith("mlp.gate.weight") or "router" in name):
                p.mul_(a.router_scale)
                n_router += 1
            if any(z in name for z in (a.zero or [])):   # diagnostic: remove one part's contribution on both sides
                p.zero_()
                continue
            if a.same_experts and p.dim() == 3 and ".experts." in name:   # diagnostic: every expert = expert 0
                p.copy_(p[0:1].expand_as(p))
            if p.dim() == 1:   # gammas off 1 (a missing / extra +1 shows), biases off 0 (a dropped bias shows)
                p.add_(torch.randn(p.shape, generator=gen) * 0.1)
                n_norm += 1
            if q8_grid_(p):
                n_grid += 1
            else:              # the rest on the fp16 grid: exact whether the converter writes it f16 or f32
                p.data.copy_(p.data.half().float())
    m.save_pretrained(hf, safe_serialization=True)
    tok.save_pretrained(hf)
    print(f"built {a.arch}: {sum(p.numel() for p in m.parameters()) / 1e6:.1f} M params, {n_grid} tensors on the Q8_0 grid, "
          f"{n_norm} 1-D tensors jittered, {n_router} routers x {a.router_scale}, the rest on the fp16 grid; transformers {transformers.__version__}")
    env = dict(os.environ, PYTHONPATH=os.path.join(LLAMA, "gguf-py") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    gg = os.path.join(a.out, "mini.gguf")
    r = subprocess.run([sys.executable, os.path.join(LLAMA, "convert_hf_to_gguf.py"), hf, "--outfile", gg,
                        "--outtype", a.outtype], env=env, capture_output=True, text=True)
    open(os.path.join(a.out, "convert.log"), "w").write(r.stdout + r.stderr)
    if r.returncode:
        print(f"convert FAILED (rc {r.returncode}): {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
        return 1
    json.dump(dict(arch=a.arch, vocab=vocab, seed=a.seed, overrides=over, outtype=a.outtype),
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"converted: {gg} ({os.path.getsize(gg) / 2**20:.1f} MiB)")
    return 0


def cmd_requant(a):
    """A built mini re-quantized by llama-quantize to any llama.cpp type (Q4_K_M, Q6_K, IQ4_XS, ...): the reference is then
    transformers loading THAT GGUF (its gguf_file loader dequantizes every tensor), so the engine's K-quant / IQ kernels are
    checked against the exact weights. Needs an arch transformers' GGUF loader knows (llama, qwen2, qwen3, qwen3_moe, phi3,
    gemma2/3/4, gpt_oss, ...); qwen35 / qwen3next are not among them -- they stay on the Q8_0-grid build."""
    q = os.environ.get("LLAMA_QUANTIZE", os.path.join(LLAMA, "build-vk", "bin", "llama-quantize"))
    dst = a.dst or f"{a.out.rstrip('/')}-{a.type}"
    os.makedirs(dst, exist_ok=True)
    r = subprocess.run([q, "--allow-requantize", os.path.join(a.out, "mini.gguf"), os.path.join(dst, "mini.gguf"), a.type],
                       capture_output=True, text=True)
    open(os.path.join(dst, "quantize.log"), "w").write(r.stdout + r.stderr)
    if r.returncode:
        print(f"llama-quantize FAILED (rc {r.returncode}): {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
        return 1
    meta = json.load(open(os.path.join(a.out, "meta.json")))
    meta.update(ref="gguf", outtype=a.type, requant_of=os.path.abspath(a.out))
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=1)
    print(f"requantized: {dst}/mini.gguf ({os.path.getsize(os.path.join(dst, 'mini.gguf')) / 2**20:.1f} MiB, {a.type})")
    return 0


def cmd_run(a):
    ie = os.environ.get("IE_BIN", os.path.join(ROOT, "build", "src", "ie"))
    dump = os.path.join(a.out, "dump")
    for ext in (".prompt.i32", ".logits.f16", ".picks.i32"):
        if os.path.exists(dump + ext):
            os.remove(dump + ext)
    env = dict(os.environ, IE_LOGITS_DUMP=dump)
    log = open(os.path.join(a.out, "serve.log"), "w")
    sp = subprocess.Popen([ie, "serve", os.path.join(a.out, "mini.gguf"), "--gpus", "1", "--cards", str(a.gpu),
                           "--parallel", "1", "--ctx", "2048", "--port", str(a.port)], env=env, stdout=log, stderr=log)
    try:
        url = f"http://127.0.0.1:{a.port}"
        for _ in range(180):
            if sp.poll() is not None:
                print(f"serve exited rc {sp.returncode}: see {a.out}/serve.log")
                return 1
            try:
                urllib.request.urlopen(url + "/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        # the chat route (the only one): the dump records the exact ids the engine rendered, the reference gets the same
        body = json.dumps(dict(messages=[{"role": "user", "content": a.prompt}], max_tokens=a.tokens, temperature=0,
                               ignore_eos=True)).encode()
        req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
        json.dump(resp, open(os.path.join(a.out, "reply.json"), "w"), indent=1)
        print(f"engine: {resp.get('usage')}")
    finally:
        sp.send_signal(signal.SIGTERM)
        try:
            sp.wait(60)
        except subprocess.TimeoutExpired:
            sp.kill()
    return 0 if os.path.exists(dump + ".logits.f16") else 1


def cmd_compare(a):
    import torch
    from transformers import AutoModelForCausalLM
    meta = json.load(open(os.path.join(a.out, "meta.json")))
    dump = os.path.join(a.out, "dump")
    prompt = np.fromfile(dump + ".prompt.i32", dtype=np.int32)
    picks = np.fromfile(dump + ".picks.i32", dtype=np.int32)
    eng = np.fromfile(dump + ".logits.f16", dtype=np.float16).astype(np.float32)
    V = eng.size // len(picks)
    eng = eng.reshape(len(picks), V)
    if meta.get("ref") == "gguf":   # a requant: the reference is transformers' dequant of the engine's own file
        # the build's own config (a config rebuilt from GGUF keys can drift, e.g. qwen3_moe norm_topk_prob defaults False)
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(os.path.join(meta["requant_of"], "hf"))
        m = AutoModelForCausalLM.from_pretrained(a.out, gguf_file="mini.gguf", config=cfg, dtype=torch.float32)
    else:
        m = AutoModelForCausalLM.from_pretrained(os.path.join(a.out, "hf"), dtype=torch.float32)
    m.eval()
    if a.negative:
        # NEGATIVE CONTROL: the reference with one middle layer's 1-D gammas scaled by 1.1 (a wrong norm transform in one
        # block, what a converter / loader mismatch looks like) -- the comparison MUST fail, or the gate is too weak.
        # (Scaling only input_layernorm is invisible on qwen3: q_norm / k_norm re-normalize q and k, only v moves.)
        L = m.config.num_hidden_layers // 2
        gs = [p for n, p in m.named_parameters() if p.dim() == 1 and f".{L}." in n]
        with torch.no_grad():
            for g in gs:
                g.mul_(1.1)
        print(f"negative control: {len(gs)} gammas of layer {L} scaled by 1.1")
    ids = torch.tensor(np.concatenate([prompt, picks[:-1]]).astype(np.int64))[None]
    with torch.no_grad():
        ref = m(ids).logits[0].numpy()
    P = len(prompt)
    rows = []
    for j in range(len(picks)):
        r, e = ref[P - 1 + j][:V], eng[j]
        cos = float(r @ e / (np.linalg.norm(r) * np.linalg.norm(e) + 1e-30))
        top2 = np.sort(r)[-2:]
        decisive = (top2[1] - top2[0]) > a.argmax_margin * (np.std(r) + 1e-30)
        rows.append((cos, int(np.argmax(r) == np.argmax(e) or not decisive), float(np.max(np.abs(r - e)) / (np.std(r) + 1e-30))))
    if a.steps:
        print("step  1-cos      max|d|/std  argmax-ok")
        for j, (cos, okm, d) in enumerate(rows):
            print(f"{j:4d}  {1 - cos:.2e}  {d:9.4f}  {okm}")
    cmin = min(r[0] for r in rows)
    dmin = min((r[0] for r in rows[1:]), default=1.0)
    am = sum(r[1] for r in rows)
    ok = rows[0][0] >= a.min_cos and dmin >= a.decode_cos and am >= len(rows) - a.argmax_slack
    print(f"[{meta['arch']}] {len(rows)} steps (prompt {P}): step-0 (prefill) cos {rows[0][0]:.6f}, min cos {cmin:.6f}, "
          f"max |d|/std {max(r[2] for r in rows):.4f}, argmax {am}/{len(rows)}")
    if a.sensitivity:
        # DIAGNOSTIC, not a gate (a planted 10 % norm error can sit inside the same envelope): how far do the REFERENCE's own
        # rows move when every MoE block's input gets Gaussian noise of relative size eps? An engine path that rounds
        # activations to 8 bits behaves like eps ~ 0.3-1 %; a random router flips experts under it, hence single-step spikes.
        blocks = [mod for mod in m.modules() if hasattr(mod, "experts") and (hasattr(mod, "gate") or hasattr(mod, "router"))]
        if not blocks:
            print("sensitivity: no MoE block found in the reference (nothing to perturb)")
        else:
            eng1 = [1 - r[0] for r in rows]
            print(f"sensitivity ({len(blocks)} MoE blocks)   median 1-cos   max 1-cos")
            print(f"  engine vs reference          {float(np.median(eng1)):.2e}     {max(eng1):.2e}")
            st = {"eps": 0.0, "seed": 0}
            def pre(mod, args):
                x = args[0]
                g = torch.Generator().manual_seed(st["seed"])
                return (x + torch.randn(x.shape, generator=g) * st["eps"] * x.pow(2).mean(-1, keepdim=True).sqrt(),) + tuple(args[1:])
            hs = [b.register_forward_pre_hook(pre) for b in blocks]
            for eps in (3e-3, 1e-2):
                res = []
                for seed in range(3):
                    st.update(eps=eps, seed=seed)
                    with torch.no_grad():
                        nz = m(ids).logits[0].numpy()
                    for j in range(len(picks)):
                        x, y = ref[P - 1 + j][:V], nz[P - 1 + j][:V]
                        res.append(1 - float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30)))
                print(f"  reference + {100 * eps:.1f} % input noise  {float(np.median(res)):.2e}     {max(res):.2e}")
            for h in hs: h.remove()
    tag = f"{meta['arch']}{' NEGATIVE CONTROL' if a.negative else ''}"
    print(f"[{tag}] {'PASS' if ok else 'FAIL'} (prefill row cos >= {a.min_cos}, decode rows >= {a.decode_cos}, "
          f"argmax misses <= {a.argmax_slack})")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--arch", required=True, help="a transformers model_type (llama, qwen3, qwen3_moe, ...)")
    b.add_argument("--out", required=True)
    b.add_argument("--layers", type=int, default=4)
    b.add_argument("--set", action="append", help="config override key=value (JSON values), repeatable")
    b.add_argument("--tokenizer-dir", required=True, help="a dir with tokenizer.json (+ tokenizer_config.json)")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--outtype", default="q8_0")
    b.add_argument("--router-scale", type=float, default=1.0, help="multiply MoE router weights (a peaked softmax: a swap at the k-th place weighs little)")
    b.add_argument("--zero", action="append", help="diagnostic: zero every parameter whose name contains this (repeatable)")
    b.add_argument("--same-experts", action="store_true", help="diagnostic: every routed expert a copy of expert 0 "
                   "(which experts are picked stops mattering; a pass then = the routing differs, a fail = elsewhere)")
    q = sub.add_parser("requant")
    q.add_argument("--out", required=True, help="a built mini")
    q.add_argument("--type", required=True, help="a llama-quantize type: Q4_K_M, Q5_K_M, Q6_K, IQ4_XS, ...")
    q.add_argument("--dst", help="default: <out>-<type>")
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--gpu", type=int, default=0)
    r.add_argument("--port", type=int, default=11593)
    r.add_argument("--tokens", type=int, default=16)
    r.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. In 1969, the first people walked on the")
    c = sub.add_parser("compare")
    c.add_argument("--out", required=True)
    # 0.9999: the weights are exact (Q8_0 grid both sides) and the model is shallow, so the engine sits near 1 - cos 4e-6;
    # 0.999 let the negative control through (a 10 % norm error in one of four blocks measured 1 - cos 1.9e-4)
    c.add_argument("--min-cos", type=float, default=0.9999, help="the prefill row (step 0)")
    # decode rows: 0.999, the qwen4exp oracle's decode bar -- the int8-activation decode kernels (default on the qwen35
    # split path) measured 1 - cos 5..7e-4, flat over the steps; the A16 path 1.3e-4; plain dense 4e-6
    c.add_argument("--decode-cos", type=float, default=0.999, help="every decode row (steps 1..)")
    c.add_argument("--argmax-margin", type=float, default=0.05)
    c.add_argument("--argmax-slack", type=int, default=1)
    c.add_argument("--negative", action="store_true", help="the negative control: a perturbed reference that must FAIL")
    c.add_argument("--steps", action="store_true", help="print every step's 1 - cos")
    c.add_argument("--sensitivity", action="store_true", help="diagnostic: the reference's own response to noise on its MoE inputs "
                   "(tells an 8-bit activation path's rounding from a defect; not a pass criterion)")
    a = ap.parse_args()
    return {"build": cmd_build, "requant": cmd_requant, "run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
