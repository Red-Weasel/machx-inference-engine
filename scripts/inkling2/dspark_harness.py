# Acceptance harness for Inkling 2 Mini's draft head (mtp.safetensors, the "DSpark" drafter: 3 dense layers over
# DFlare-mixed target features + a recurrent rnn_hidden Markov head), on the CPU in fp32.
#
#   <ref venv>/bin/python -I dspark_harness.py MODEL_DIR FEAT_DIR [FEAT_DIR ...] [--out drafts.json] [--variant NAME]
#
# FEAT_DIR: ie-inkling2-run --generate N --dump-features FEAT_DIR (features.f32 [positions][taps][hidden], tokens.json).
# For every anchor position p in the generated part (the anchor = the token at p, the context = the features of
# positions < p), the head drafts block_size - 1 tokens greedily; a draft is accepted while it equals the target's own
# greedy continuation. Prints the mean accepted length per verify and the histogram; --out writes the drafts (the
# engine port's golden).
#
# The forward follows vLLM's Inkling2 DSpark (vllm/models/inkling2/nvidia/dspark.py + speculator.py):
#   features: taps at visits target_layer_ids[:-1] = the partial state entering that layer call, the last tap = the
#     final-norm output; fuse_alpha = softmax(fuse_weight) per row (rows 0..L-1 the draft layers, row L the Markov head)
#   context of layer i: f = RMSNorm(sum_j alpha_ij tap_j) * layers.i.dflare_norm; K = rope(k_norm_ctx(k_proj_ctx f)) at the
#     context position, V = v_proj_ctx f; local layers (local_layer_ids) see the sliding_window_size positions before
#     the block start, the others all of them
#   block (DFlash 1 + N layout): row 0 = embed_norm(embed[anchor]) at position p, rows 1..B-1 = mask_embed at p + k;
#     pre-norm layers, q / k per-head RMSNorm then neox RoPE (theta from mtp_config), attention over [context, block]
#     without a causal mask, SiLU MLP; final norm
#   row k (k >= 1) drafts the token at p + k: base = unembed(h_k); Markov step: z = [state, w1[prev], h_k, f_rnn]
#     (f_rnn = the Markov mix of the features at p - 1), raw = rnn_proj z -> gate / cand / out; state = sig(gate) * state
#     + (1 - sig(gate)) * tanh(cand); bias = unembed(hidden_out(tanh(out))); logits = softcap(base + bias)
import sys, os, json, argparse, numpy as np, torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("feats", nargs="+")
ap.add_argument("--out")
ap.add_argument("--variant", default="ref")
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--max-anchors", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(a.threads)
cfg = json.load(open(os.path.join(a.model, "config.json")))
mc = cfg["mtp_config"]
tc = cfg.get("text_config", cfg)
H, NH, NKV, D = mc["hidden_size"], mc["num_attention_heads"], mc["num_key_value_heads"], mc["head_dim"]
L, B, R, eps = mc["num_hidden_layers"], mc["block_size"], mc["markov_rank"], mc["rms_norm_eps"]
WIN, local = mc["sliding_window_size"], set(mc["local_layer_ids"])
softcap = tc.get("final_logit_softcapping", 30.0)
V = 200058   # the engine's unpadded vocabulary (logits beyond it are never picked)

def ld(f, name): return f.get_tensor(name).float()
mt = safe_open(os.path.join(a.model, "mtp.safetensors"), "pt")
W = {k: mt.get_tensor(k).float() for k in mt.keys()}
mm = safe_open(os.path.join(a.model, "model-00001-of-00001.safetensors"), "pt")
embed = mm.get_tensor("model.llm.embed.weight")                      # bf16, rows indexed
embed_norm = ld(mm, "model.llm.embed_norm.weight")
unembed = ld(mm, "model.llm.unembed.weight")[:V].contiguous()

def rms(x, w): return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w
alpha = torch.softmax(W["fuse_weight"], -1)                          # [L + 1, taps]
inv = 1.0 / (mc["rope_theta"] ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
def rope(x, pos):                                                    # x [T, heads, D], neox halves
    ang = torch.tensor(pos, dtype=torch.float64)[:, None] * inv[None]
    c, s = ang.cos().float()[:, None], ang.sin().float()[:, None]
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)
hw, hb = W["markov_head.hidden_out.weight"], W["markov_head.hidden_out.bias"]
fold_w = unembed @ hw                                                # [V, R]
fold_b = unembed @ hb
w1 = W["markov_head.markov_w1.weight"]
rp_w, rp_b = W["markov_head.rnn_proj.weight"], W["markov_head.rnn_proj.bias"]

tot_acc, tot_n, hist = 0, 0, np.zeros(B, np.int64)
golden = []
for fd in a.feats:
    meta = json.load(open(os.path.join(fd, "tokens.json")))
    ids, taps, P0 = meta["ids"], len(meta["taps"]), meta["prompt_len"]
    feats = torch.from_numpy(np.fromfile(os.path.join(fd, "features.f32"), np.float32).reshape(-1, taps, H))
    N = feats.shape[0]                                               # positions with features (ids has N + 1)
    ctx = []
    for i in range(L):                                               # the context K / V of every position, per layer
        f = rms((feats * alpha[i][None, :, None]).sum(1), W[f"layers.{i}.dflare_norm.weight"])
        p = f"layers.{i}.self_attn."
        k = (f @ W[p + "k_proj_ctx.weight"].T).view(N, NKV, D)
        k = rope(rms(k, W[p + "k_norm_ctx.weight"]), list(range(N)))
        v = (f @ W[p + "v_proj_ctx.weight"].T).view(N, NKV, D)
        ctx.append((k, v))
    fr_all = rms((feats * alpha[L][None, :, None]).sum(1), W["markov_head.dflare_norm.weight"])
    anchors = [p for p in range(P0, len(ids) - B + 1)]
    if a.max_anchors: anchors = anchors[: a.max_anchors]
    acc_f = 0
    for p in anchors:
        x = W["mask_embed"].expand(B, H).clone()
        x[0] = rms(embed[ids[p]].float(), embed_norm)
        pos = list(range(p, p + B))
        for i in range(L):
            pre = f"layers.{i}."
            xn = rms(x, W[pre + "input_layernorm.weight"])
            q = rope(rms((xn @ W[pre + "self_attn.q_proj.weight"].T).view(B, NH, D), W[pre + "self_attn.q_norm.weight"]), pos)
            k = rope(rms((xn @ W[pre + "self_attn.k_proj.weight"].T).view(B, NKV, D), W[pre + "self_attn.k_norm.weight"]), pos)
            v = (xn @ W[pre + "self_attn.v_proj.weight"].T).view(B, NKV, D)
            lo = max(0, p - WIN) if i in local else 0
            kc, vc = ctx[i]
            K = torch.cat([kc[lo:p], k], 0).repeat_interleave(NH // NKV, 1)      # [S, NH, D]
            Vv = torch.cat([vc[lo:p], v], 0).repeat_interleave(NH // NKV, 1)
            s = torch.einsum("thd,shd->hts", q, K) / D ** 0.5
            o = torch.einsum("hts,shd->thd", torch.softmax(s, -1), Vv).reshape(B, NH * D)
            x = x + o @ W[pre + "self_attn.o_proj.weight"].T
            xn = rms(x, W[pre + "post_attention_layernorm.weight"])
            g, u = xn @ W[pre + "mlp.gate_proj.weight"].T, xn @ W[pre + "mlp.up_proj.weight"].T
            x = x + (torch.nn.functional.silu(g) * u) @ W[pre + "mlp.down_proj.weight"].T
        h = rms(x, W["norm.weight"])
        base = h[1:] @ unembed.T                                     # [B - 1, V]
        fr = fr_all[p - 1]
        ctx_term = torch.cat([h[1:], fr.expand(B - 1, H)], -1) @ rp_w[:, 2 * R :].T + rp_b   # [B - 1, 3R]
        state, prev, drafts = torch.zeros(R), ids[p], []
        for i in range(B - 1):
            raw = ctx_term[i] + torch.cat([state, w1[prev]]) @ rp_w[:, : 2 * R].T
            gate, cand, out = torch.sigmoid(raw[:R]), torch.tanh(raw[R : 2 * R]), raw[2 * R :]
            state = gate * state + (1 - gate) * cand
            lat = torch.tanh(out)
            if a.variant == "nomarkov": lg = base[i]
            else: lg = base[i] + fold_w @ lat + fold_b
            lg = softcap * torch.tanh(lg / softcap)
            prev = int(lg.argmax())
            drafts.append(prev)
        n = 0
        while n < B - 1 and drafts[n] == ids[p + 1 + n]: n += 1
        hist[n] += 1; tot_acc += n; tot_n += 1; acc_f += n
        golden.append({"feat": fd, "anchor": p, "drafts": drafts, "accepted": n})
    print(f"{os.path.basename(fd)}: {len(anchors)} anchors, mean accepted {acc_f / max(1, len(anchors)):.2f} of {B - 1}", flush=True)
print(f"ALL ({a.variant}): {tot_n} anchors, mean accepted {tot_acc / max(1, tot_n):.3f} -> {1 + tot_acc / max(1, tot_n):.2f} tokens per verify; "
      f"histogram {hist.tolist()}")
if a.out: json.dump(golden, open(a.out, "w"))
