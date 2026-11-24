#!/usr/bin/env python3
"""GPTQ for Xing4.0-29B-A4B into llama.cpp's own Q6_K blocks (the file stays a normal GGUF the engine and llama.cpp read).

  python gptq.py --model <HF dir> --template <plain .gguf> --out <new .gguf> [--nseq 128] [--seqlen 512] [--threads 16]
                 [--work <dir>] [--damp 0.01] [--layers 0-39] [--no-head]

Sequential, layer by layer, on the CPU with the release's own modeling code (fp32 math):
  1. calibration text (the repo's docs, its C++ sources, chat prompts in the chat template) -> --nseq sequences
  2. per layer: run the layer on the current residual streams with hooks that sum X^T X for every matrix input (each
     routed expert sees only the tokens routed to it), GPTQ-round every matrix the template stores as Q6_K onto the Q6_K
     grid (per 16 weights an int8 scale, per 256 an fp16 d; errors pushed onto the later columns along the Cholesky
     factor of H^-1), write the blocks over the template's bytes, put the rounded weights into the layer, and run the
     layer again for the next layer's input
  3. the head (lm_head) the same way on the final norm's output
Tensors the template stores in another type keep the template's bytes. The packed blocks are checked against gguf-py's
own Q6_K dequantizer before they are written. --work keeps the streams after each layer (restart with --layers N-39).
"""
import argparse, glob, json, os, re, shutil, sys, time
import numpy as np
import torch

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ------------------------------------------------------------------------------------------------ calibration text
def agent_texts(tok, files):
    """Agent conversations rendered through the release's template with their tools: lines of {"messages","tools"}
    (agentgen.py) or ie serve's IE_SERVE_RECORD lines ({"request": {...}, "reply": {...}}, the reply as the last turn)."""
    out = []
    for fn in files:
        for line in open(fn, errors="ignore"):
            try: j = json.loads(line)
            except ValueError: continue
            if "request" in j:
                rq = j["request"] if isinstance(j["request"], dict) else {}
                msgs = list(rq.get("messages", [])); tools = rq.get("tools")
                rp = j.get("reply", {})
                last = {"role": "assistant", "content": rp.get("content", "")}
                if rp.get("reasoning"): last["reasoning_content"] = rp["reasoning"]
                tc = rp.get("tool_calls")
                if isinstance(tc, str):
                    try: tc = json.loads(tc)
                    except ValueError: tc = None
                if tc: last["tool_calls"] = tc
                msgs.append(last)
            else:
                msgs = j.get("messages", []); tools = j.get("tools")
            for m in msgs:   # the template writes arguments item by item: it needs a mapping, OpenAI sends a JSON string
                if not isinstance(m.get("content"), str): m["content"] = "" if m.get("content") is None else json.dumps(m["content"])
                for c in m.get("tool_calls") or []:
                    f = c.get("function", c)
                    if isinstance(f.get("arguments"), str):
                        try: f["arguments"] = json.loads(f["arguments"])
                        except ValueError: f["arguments"] = {}
            try: out.append(tok.apply_chat_template(msgs, tools=tools or None, tokenize=False))
            except Exception as e: print("agent text skipped:", e, flush=True)
    return out

def calib_ids(tok, nseq, seqlen, agent_files=()):
    def stream(texts):
        ids = []
        for t in texts:
            ids += tok(t, add_special_tokens=False)["input_ids"]
        return ids
    docs = []
    for f in sorted(glob.glob(os.path.join(REPO, "docs", "**", "*.md"), recursive=True)) + [os.path.join(REPO, "README.md")]:
        try: docs.append(open(f, errors="ignore").read())
        except OSError: pass
    code = []
    for f in sorted(glob.glob(os.path.join(REPO, "src", "**", "*.cpp"), recursive=True))[::3]:
        code.append(open(f, errors="ignore").read()[:20000])
    chat = []
    data = os.path.expanduser("~/models/hy4flash-ablit/data/harmless.json")
    if os.path.exists(data):
        for p in json.load(open(data))[:600]:
            msgs = [{"role": "user", "content": p}]
            chat.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    agent = agent_texts(tok, agent_files) if agent_files else []
    if agent:   # docs / code / chat / agent a quarter each
        q = nseq // 4
        parts = [(stream(docs), q), (stream(code), q), (stream(chat), q), (stream(agent), nseq - 3 * q)]
        print(f"calibration: {len(agent)} agent conversations, {len(stream(agent))} tokens, as {nseq - 3 * q} of {nseq} sequences", flush=True)
    else:
        parts = [(stream(docs), nseq // 2), (stream(code), nseq // 4), (stream(chat), nseq - nseq // 2 - nseq // 4)]
    seqs = []
    for ids, n in parts:
        if len(ids) < seqlen: continue
        step = max(seqlen, (len(ids) - seqlen) // max(1, n))
        for i in range(n):
            o = (i * step) % (len(ids) - seqlen)
            seqs.append(ids[o:o + seqlen])
    while len(seqs) < nseq: seqs.append(seqs[len(seqs) % max(1, len(seqs))])
    return torch.tensor(seqs[:nseq], dtype=torch.long)

# ------------------------------------------------------------------------------------------------------ Q6_K + GPTQ
def gptq_q6k(W, H, damp):
    """W [M, N, K] fp32 (M matrices sharing nothing), H [M, K, K] fp64 (X^T X). Returns (q int8 [M,N,K] in -32..31,
    sc int8 [M,N,K/16], d fp16 [M,N,K/256], W_hat fp32)."""
    M, N, K = W.shape
    W = W.clone()
    H = H.clone()
    diag = torch.diagonal(H, dim1=1, dim2=2)
    dead = diag <= 0
    for m in range(M):
        if dead[m].any():
            H[m][dead[m], dead[m]] = 1.0
            W[m][:, dead[m]] = 0.0
    mean = torch.diagonal(H, dim1=1, dim2=2).mean(dim=1)
    H += (damp * mean).view(M, 1, 1) * torch.eye(K, dtype=H.dtype).unsqueeze(0)
    # upper factor U with H^-1 = U^T U
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    U = torch.linalg.cholesky(Hinv, upper=True).to(torch.float32)
    del L, Hinv
    q = torch.zeros((M, N, K), dtype=torch.int8)
    sc = torch.zeros((M, N, K // 16), dtype=torch.int8)
    d = torch.zeros((M, N, K // 256), dtype=torch.float16)
    Wq = torch.zeros_like(W)
    B = 128
    s_cur = None
    dsb = None
    for i1 in range(0, K, B):
        i2 = min(i1 + B, K)
        W1 = W[:, :, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Hb = U[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % 256 == 0:   # the super-block's d from the scales the current weights would want
                blkw = torch.cat([W1[:, :, i:], W[:, :, i2:col + 256]], dim=2) if col + 256 > i2 else W1[:, :, i:i + 256]
                s16 = blkw.abs().view(M, N, 16, 16).amax(dim=3) / 31.0
                dsb = (s16.amax(dim=2) / 127.0).to(torch.float16)
                dsb = torch.where(dsb == 0, torch.full_like(dsb, 1e-8), dsb)
                d[:, :, col // 256] = dsb
            if col % 16 == 0:    # this group's scale, from the weights as GPTQ has updated them
                g = W1[:, :, i:i + 16].abs().amax(dim=2) / 31.0
                si = torch.clamp(torch.round(g / dsb.float()), 1, 127)
                sc[:, :, col // 16] = si.to(torch.int8)
                s_cur = si * dsb.float()
            w = W1[:, :, i]
            qi = torch.clamp(torch.round(w / s_cur), -32, 31)
            dq = qi * s_cur
            q[:, :, col] = qi.to(torch.int8)
            Wq[:, :, col] = dq
            err = (w - dq) / Hb[:, i, i].unsqueeze(1)
            W1[:, :, i:] -= err.unsqueeze(2) * Hb[:, i, i:].unsqueeze(1)
            E1[:, :, i] = err
        if i2 < K:
            W[:, :, i2:] -= torch.bmm(E1, U[:, i1:i2, i2:])
    return q, sc, d, Wq

def pack_q6k(q, sc, d):
    """q int8 [R, K] (-32..31), sc int8 [R, K/16], d fp16 [R, K/256] -> ggml block_q6_K bytes [R, K/256 * 210]."""
    R, K = q.shape
    nb = K // 256
    u = (q.astype(np.int16) + 32).astype(np.uint8).reshape(R, nb, 2, 128)
    a, b, c, e = u[..., 0:32], u[..., 32:64], u[..., 64:96], u[..., 96:128]
    ql = np.empty((R, nb, 2, 64), np.uint8)
    ql[..., :32] = (a & 15) | ((c & 15) << 4)
    ql[..., 32:] = (b & 15) | ((e & 15) << 4)
    qh = (a >> 4) | ((b >> 4) << 2) | ((c >> 4) << 4) | ((e >> 4) << 6)
    out = np.empty((R, nb, 210), np.uint8)
    out[..., 0:128] = ql.reshape(R, nb, 128)
    out[..., 128:192] = qh.reshape(R, nb, 64)
    out[..., 192:208] = sc.reshape(R, nb, 16).view(np.uint8)
    out[..., 208:210] = d.reshape(R, nb, 1).view(np.uint8).reshape(R, nb, 2)
    return out.reshape(R, nb * 210)

# --------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--template", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--nseq", type=int, default=128); ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--threads", type=int, default=16); ap.add_argument("--work", default="")
    ap.add_argument("--damp", type=float, default=0.01); ap.add_argument("--layers", default="")
    ap.add_argument("--agent", action="append", default=[], help="agent conversations JSONL (agentgen.py or IE_SERVE_RECORD); a quarter of the calibration")
    ap.add_argument("--no-head", action="store_true"); ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, os.path.join(os.environ.get("IE_DATA_DRIVE", os.path.expanduser("~/data")), "src/llama.cpp-xing/gguf-py"))
    import gguf
    from gguf.quants import dequantize
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.masking_utils import create_causal_mask

    rd = gguf.GGUFReader(a.template)
    tinfo = {t.name: t for t in rd.tensors}
    if not os.path.exists(a.out):
        log(f"copying the template -> {a.out}")
        shutil.copyfile(a.template, a.out)
    fout = open(a.out, "r+b")
    def target(name):
        t = tinfo.get(name)
        return t if t is not None and t.tensor_type == gguf.GGMLQuantizationType.Q6_K else None
    def write(name, q, sc, d, wq_ref):
        t = tinfo[name]
        R = q.shape[0]
        blocks = pack_q6k(q, sc, d)
        assert blocks.nbytes == t.n_bytes, (name, blocks.nbytes, t.n_bytes)
        back = dequantize(blocks[: min(R, 64)], gguf.GGMLQuantizationType.Q6_K)
        ref = wq_ref[: min(R, 64)]
        if not np.allclose(back, ref, rtol=1e-3, atol=1e-6 * max(1e-3, float(np.abs(ref).max()))):
            raise RuntimeError(f"{name}: packed blocks do not dequantize to the rounded weights (max diff {np.abs(back - ref).max()})")
        fout.seek(t.data_offset); fout.write(blocks.tobytes())

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    ids = calib_ids(tok, a.nseq, a.seqlen, a.agent)
    log(f"calibration: {ids.shape[0]} sequences x {ids.shape[1]} tokens")
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    cfg = model.config
    mm = model.model
    NL = cfg.num_hidden_layers
    lay = range(NL)
    if a.layers:
        lo, hi = (int(x) for x in a.layers.split("-")); lay = range(lo, hi + 1)
    T = a.seqlen
    pos = torch.arange(T).unsqueeze(0)
    work = a.work or os.path.dirname(os.path.abspath(a.out))
    sfile = lambda L: os.path.join(work, f"xg4_gptq_streams_L{L:02d}.pt")
    with torch.no_grad():
        if lay[0] == 0:
            X = torch.empty((ids.shape[0], T, cfg.hc_mult, cfg.hidden_size), dtype=torch.float32)
            for i in range(ids.shape[0]):
                X[i] = mm.embed_tokens(ids[i:i + 1]).float()[0].unsqueeze(1).expand(-1, cfg.hc_mult, -1)
        else:
            X = torch.load(sfile(lay[0]))
        emb0 = torch.zeros((1, T, cfg.hidden_size))
        mask = create_causal_mask(config=cfg, inputs_embeds=emb0, attention_mask=None, past_key_values=None, position_ids=pos)
        pe = mm.rotary_emb(X[:1].float(), position_ids=pos)
        pe = tuple(p.float() for p in pe)

        def run_layer(layer, Xin, hooks=None):
            out = torch.empty_like(Xin)
            for s in range(0, Xin.shape[0], a.batch):
                xb = Xin[s:s + a.batch]
                pb = pos.expand(xb.shape[0], -1)
                mb = mask if mask is None else mask.expand(xb.shape[0], *mask.shape[1:])
                out[s:s + a.batch] = layer(xb, attention_mask=mb, position_embeddings=pe, position_ids=pb)
            return out

        for L in lay:
            t0 = time.time()
            layer = mm.layers[L].float()
            at = layer.self_attn
            mods = {}   # input key -> (module that sees it, [(hf linear, gguf name, row slice / None)])
            p = f"blk.{L}."
            mods["attn_in"] = (at.q_a_proj, [(at.q_a_proj, p + "attn_q_a.weight"), (at.kv_a_proj_with_mqa, p + "attn_kv_a_mqa.weight")])
            mods["q_b"] = (at.q_b_proj, [(at.q_b_proj, p + "attn_q_b.weight")])
            mods["kv_b"] = (at.kv_b_proj, [(at.kv_b_proj, p + "attn_v_b.weight")])
            mods["o"] = (at.o_proj, [(at.o_proj, p + "attn_output.weight")])
            moe = hasattr(layer.mlp, "experts")
            if moe:
                se = layer.mlp.shared_experts
                mods["sh_in"] = (se.gate_proj, [(se.gate_proj, p + "ffn_gate_shexp.weight"), (se.up_proj, p + "ffn_up_shexp.weight")])
                mods["sh_dn"] = (se.down_proj, [(se.down_proj, p + "ffn_down_shexp.weight")])
            else:
                ml = layer.mlp
                mods["ffn_in"] = (ml.gate_proj, [(ml.gate_proj, p + "ffn_gate.weight"), (ml.up_proj, p + "ffn_up.weight")])
                mods["ffn_dn"] = (ml.down_proj, [(ml.down_proj, p + "ffn_down.weight")])
            Hs, hk = {}, []
            def acc(key):
                def f(mod, inp, out):
                    x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                    h = (x.T @ x).double()
                    if key in Hs: Hs[key] += h
                    else: Hs[key] = h
                return f
            for key, (m, _) in mods.items(): hk.append(m.register_forward_hook(acc(key)))
            if moe:
                for e, ex in enumerate(layer.mlp.experts):
                    hk.append(ex.gate_proj.register_forward_hook(acc(("e_in", e))))
                    hk.append(ex.down_proj.register_forward_hook(acc(("e_dn", e))))
            run_layer(layer, X)
            for h in hk: h.remove()
            t1 = time.time()
            # dense / attention matrices
            for key, (_, lst) in mods.items():
                H = Hs[key].unsqueeze(0)
                for lin, name in lst:
                    W = lin.weight.data.float()
                    rows = None
                    if name.endswith("attn_v_b.weight"):
                        nh, nope, vd = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.v_head_dim
                        rows = torch.cat([torch.arange(h * (nope + vd) + nope, (h + 1) * (nope + vd)) for h in range(nh)])
                        W = W[rows]
                    if target(name) is None:
                        continue
                    q, sc, d, Wq = gptq_q6k(W.unsqueeze(0), H, a.damp)
                    write(name, q[0].numpy(), sc[0].numpy(), d[0].numpy(), Wq[0].numpy())
                    if rows is None: lin.weight.data.copy_(Wq[0])
                    else: lin.weight.data[rows] = Wq[0]
            # routed experts, batched over the experts
            if moe:
                E = len(layer.mlp.experts)
                dim, inter = cfg.hidden_size, cfg.moe_intermediate_size
                for kind, gname, attr, Kk in (("e_in", p + "ffn_gate_exps.weight", "gate_proj", dim),
                                              ("e_in", p + "ffn_up_exps.weight", "up_proj", dim),
                                              ("e_dn", p + "ffn_down_exps.weight", "down_proj", inter)):
                    if target(gname) is None: continue
                    Hm = torch.stack([Hs.get((kind, e), torch.zeros((Kk, Kk), dtype=torch.float64)) for e in range(E)])
                    allq, allsc, alld = [], [], []
                    CH = 16 if Kk > 2048 else 64
                    for e0 in range(0, E, CH):
                        e1 = min(E, e0 + CH)
                        W = torch.stack([getattr(layer.mlp.experts[e], attr).weight.data.float() for e in range(e0, e1)])
                        # an expert no calibration token reached: H from the pooled statistics of the layer
                        Hc = Hm[e0:e1].clone()
                        pooled = Hm.sum(0) / max(1, E)
                        for j in range(e1 - e0):
                            if torch.diagonal(Hc[j]).sum() == 0: Hc[j] = pooled
                        q, sc, d, Wq = gptq_q6k(W, Hc, a.damp)
                        for j, e in enumerate(range(e0, e1)): getattr(layer.mlp.experts[e], attr).weight.data.copy_(Wq[j])
                        allq.append(q.reshape(-1, Kk)); allsc.append(sc.reshape(-1, Kk // 16)); alld.append(d.reshape(-1, Kk // 256))
                        if e0 == 0: wref = Wq.reshape(-1, Kk)[:64].numpy()
                    write(gname, torch.cat(allq).numpy(), torch.cat(allsc).numpy(), torch.cat(alld).numpy(),
                          np.concatenate([wref, np.zeros((0, Kk), np.float32)]))
            t2 = time.time()
            del Hs
            X = run_layer(layer, X)
            mm.layers[L] = layer.to(torch.bfloat16)
            fout.flush()
            if a.work:
                torch.save(X, sfile(L + 1))
                if os.path.exists(sfile(L)): os.remove(sfile(L))     # keep only the latest (3.8 GB each)
            log(f"layer {L}: statistics {t1 - t0:.0f} s, GPTQ {t2 - t1:.0f} s, next input {time.time() - t2:.0f} s")
        if not a.no_head and lay[-1] == NL - 1 and target("output.weight") is not None:
            t0 = time.time()
            xs = mm.norm(X.mean(dim=2)).reshape(-1, cfg.hidden_size).float()
            H = (xs.T @ xs).double().unsqueeze(0)
            W = model.lm_head.weight.data.float()
            qs_, scs, ds, wr = [], [], [], None
            for r0 in range(0, W.shape[0], 16384):
                q, sc, d, Wq = gptq_q6k(W[r0:r0 + 16384].unsqueeze(0), H, a.damp)
                qs_.append(q[0]); scs.append(sc[0]); ds.append(d[0])
                if wr is None: wr = Wq[0][:64].numpy()
            write("output.weight", torch.cat(qs_).numpy(), torch.cat(scs).numpy(), torch.cat(ds).numpy(), wr)
            log(f"head: {time.time() - t0:.0f} s")
    fout.close()
    log("done")

if __name__ == "__main__":
    main()
