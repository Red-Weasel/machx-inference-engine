#!/usr/bin/env python3
"""Hy4-flash DFly2 drafter: an independent fp32 forward of the checkpoint's drafter over inputs the engine dumped.

  python dfly_oracle.py <source_dir> <dump_dir>

The engine (IE_HY4F_DFLY_DUMP=<dump_dir>, ie-hy4flash-run --draft-eval or --draft) writes, for its first drafter calls,
the target's final hidden of the last accepted row (th), the rope table of the block's positions (cs), the target-layer
key latents each draft layer read (keysL) and its own intermediates (x0 after the fusion, xL after layer L, h, the
logits of every draft). This recomputes all of it from the BF16 weights in fp32 (reference: vLLM
models/hy_v4/nvidia/dfly*.py, AngelSpec models/draft/hy4_mla.py) and reports where the two differ.
"""
import glob, json, os, sys
import numpy as np, torch
from safetensors import safe_open

H, KL, RD, ND, VD, NH, QL, MASK = 3072, 512, 64, 192, 256, 32, 1024, 120009
EPS = 1e-5

def load(src, names):
    out = {}
    for fn in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with safe_open(fn, "pt") as f:
            for k in f.keys():
                if k in names: out[k] = f.get_tensor(k).float()
    missing = [n for n in names if n not in out]
    if missing: raise SystemExit(f"missing tensors: {missing[:4]}")
    return out

def rms(x, w): return w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)

def rope(x, cs):            # x [..., RD] interleaved pairs, cs [RD] = cos[RD/2] | sin[RD/2]
    c, s = cs[..., : RD // 2], cs[..., RD // 2:]
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x0 * c - x1 * s, x1 * c + x0 * s], dim=-1).flatten(-2)

def main():
    src, dd = sys.argv[1], sys.argv[2]
    torch.set_grad_enabled(False)
    P = "model.dfly_layers."
    names = {P + n for n in ["final_norm.weight", "mtp_fusion.eh_proj.weight", "mtp_fusion.enorm.weight", "mtp_fusion.gate", "mtp_fusion.hnorm.weight",
                             "hidden_correction.down_proj.weight", "hidden_correction.embed_norm.weight", "hidden_correction.gate_proj.weight",
                             "hidden_correction.hidden_norm.weight", "hidden_correction.up_proj.weight"]}
    for i in range(5):
        for n in ["input_layernorm.weight", "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "post_attention_layernorm.weight", "self_attn.kv_a_layernorm.weight",
                  "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_b_proj.weight", "self_attn.learnable_sink_param", "self_attn.linear_gate.weight", "self_attn.o_proj.weight",
                  "self_attn.post_proj_norm.weight", "self_attn.q_a_layernorm.weight", "self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight"]:
            names.add(f"{P}{i}.{n}")
    names |= {"model.embed_tokens.weight", "lm_head.weight"}
    w = load(src, names)
    emb, head = w["model.embed_tokens.weight"], w["lm_head.weight"]
    def rel(a, b): return float((a - b).norm() / (b.norm() + 1e-30))
    n = 0
    while os.path.exists(os.path.join(dd, f"c{n}.meta.json")):
        pre = os.path.join(dd, f"c{n}.")
        meta = json.load(open(pre + "meta.json")); W = meta["W"]
        f32 = lambda name, shape: torch.from_numpy(np.fromfile(pre + name, dtype=np.float32).reshape(shape).copy())
        th, cs = f32("th.f32", (H,)), f32("cs.f32", (W, RD))
        ids = [meta["bonus"]] + [MASK] * (W - 1)
        x = emb[ids].clone()
        z = torch.cat([rms(x[0], w[P + "mtp_fusion.enorm.weight"]), rms(th, w[P + "mtp_fusion.hnorm.weight"])])
        x[0] = x[0] + w[P + "mtp_fusion.gate"] * (w[P + "mtp_fusion.eh_proj.weight"] @ z)
        line = [f"x0 {rel(f32('x0.f32', (W, H)), x):.1e}"]
        for i in range(5):
            L = f"{P}{i}."
            y = rms(x, w[L + "input_layernorm.weight"])
            q = (rms(y @ w[L + "self_attn.q_a_proj.weight"].T, w[L + "self_attn.q_a_layernorm.weight"]) @ w[L + "self_attn.q_b_proj.weight"].T).view(W, NH, ND + RD)
            q_nope, q_pe = q[..., :ND], rope(q[..., ND:], cs[:, None, :])
            kv = y @ w[L + "self_attn.kv_a_proj_with_mqa.weight"].T
            blk = torch.cat([rms(kv[:, :KL], w[L + "self_attn.kv_a_layernorm.weight"]), rope(kv[:, KL:], cs)], dim=-1)
            nh = meta["n_hist"][i]
            hist = torch.from_numpy(np.fromfile(pre + f"keys{i}.f16", dtype=np.float16).reshape(nh, KL + RD).astype(np.float32))
            lat = torch.cat([hist, blk], dim=0)                                     # [nh + W, 576]
            kvb = w[L + "self_attn.kv_b_proj.weight"].view(NH, ND + VD, KL)
            kn = torch.einsum("hdm,sm->hsd", kvb[:, :ND], lat[:, :KL])               # [NH, S, ND]
            val = torch.einsum("hdm,sm->hsd", kvb[:, ND:], lat[:, :KL])              # [NH, S, VD]
            sc = (torch.einsum("jhd,hsd->jhs", q_nope, kn) + torch.einsum("jhr,sr->jhs", q_pe, lat[:, KL:])) / 16.0
            S = lat.shape[0]
            allow = torch.arange(S)[None, :] <= (nh + torch.arange(W))[:, None]      # slot j: the history and block slots 0 .. j
            sc = sc.masked_fill(~allow[:, None, :], float("-inf"))
            sink = w[L + "self_attn.learnable_sink_param"].view(1, NH, 1)
            pr = torch.exp(sc - torch.logaddexp(torch.logsumexp(sc, dim=-1, keepdim=True), sink))
            o = torch.einsum("jhs,hsd->jhd", pr, val)
            o = o * torch.sigmoid((y @ w[L + "self_attn.linear_gate.weight"].T).view(W, NH, VD))
            a = rms(o.reshape(W, NH * VD) @ w[L + "self_attn.o_proj.weight"].T, w[L + "self_attn.post_proj_norm.weight"])
            x = x + a
            y2 = rms(x, w[L + "post_attention_layernorm.weight"])
            x = x + (torch.nn.functional.silu(y2 @ w[L + "mlp.gate_proj.weight"].T) * (y2 @ w[L + "mlp.up_proj.weight"].T)) @ w[L + "mlp.down_proj.weight"].T
            line.append(f"x{i + 1} {rel(f32(f'x{i + 1}.f32', (W, H)), x):.1e}")
        h = rms(x, w[P + "final_norm.weight"])
        line.append(f"h {rel(f32('h.f32', (W, H)), h):.1e}")
        prev, drafts, lrel = meta["bonus"], [], []
        for k in range(1, W):
            z = torch.cat([rms(h[k], w[P + "hidden_correction.hidden_norm.weight"]), rms(emb[prev], w[P + "hidden_correction.embed_norm.weight"])])
            hc = h[k] + w[P + "hidden_correction.down_proj.weight"] @ (torch.nn.functional.silu(w[P + "hidden_correction.gate_proj.weight"] @ z) * (w[P + "hidden_correction.up_proj.weight"] @ z))
            lg = head @ hc
            if os.path.exists(pre + f"logits{k}.f32") and prev == ([meta["bonus"]] + meta["drafts"])[k - 1]: lrel.append(rel(f32(f"logits{k}.f32", (lg.shape[0],)), lg))
            prev = int(lg.argmax()); drafts.append(prev)
        print(f"call {n}: engine drafts {meta['drafts']}  oracle drafts {drafts}  {'SAME' if drafts == meta['drafts'] else 'DIFFERENT'}")
        print("   relative difference, engine vs oracle: " + "  ".join(line) + "  logits " + " ".join(f"{v:.1e}" for v in lrel))
        n += 1
    if n == 0: raise SystemExit("no dumped calls found")

if __name__ == "__main__": main()
