"""Inkling 2 Mini (arch inkling2_loop) for LoRA training on one Intel Arc card.

The publisher's reference implementation (transformers 5.24.0, Inkling2LoopForConditionalGeneration) is used
UNPATCHED; everything here wraps modules from the outside:

- CompressedExperts (compressed_experts.py): the frozen experts stay in their NVFP4 / MXFP8 checkpoint form on the
  card and are decoded per call (exact fp32 amax, as the engine serves them). This is what makes the model fit 32 GiB.
- LoRALinear on the attention q/k/v/o of all 24 layers, "merged": the forward uses bf16(fp32(W) + scale * B @ A),
  exactly the tensor the export writes, so the trained model is the exported model bit for bit.
- ReplayRouter: the router's top-k picks of the forward are stored per layer CALL and reused when gradient
  checkpointing recomputes the layer in the backward. Without it the recompute can pick other experts on a near-tie
  (GPU arithmetic is not bit-reproducible), the expert index tensors change shape and torch.utils.checkpoint aborts
  (the [305] vs [306] error); with it the recompute is the forward's computation and the gradient is the forward's.
  The routing WEIGHTS are recomputed from the scores at the stored indices, so their gradient is exact.
- EngineAblation: the run-time ablation / steering file of a model folder (ie_ablation.json, format ie-ablation-v1)
  applied the way the engine applies it (src/model/inkling2.cpp, i2_ablate_add): every listed write w becomes
  w - s * sum_j (w . r_j) r_j (+ s * sum_j r_j for "add" groups) on the attention write (post-attention short conv
  output), the MoE write (post-MoE conv output) of calls 0..47, and the normed embedding. So an abliterated folder is
  trained as the model the engine serves, not as the plain checkpoint.
- Chunked attention for long examples: the reference materialises [heads, T, T] scores and position biases (6 GiB
  each at 8,192 tokens in fp32); above CHUNK_MIN tokens the same arithmetic runs per block of query rows, each block
  under its own checkpoint, with keys limited to the sliding window on the sliding layers.
- The head (lm_head + softcap) is computed only at the trained positions, in checkpointed row blocks.

Call numbering (as the engine's): layers 0..5 -> calls 0..5, loop layers 6..17 pass p (0, 1, 2) -> 6 + 12 p + (L - 6),
layers 18..23 -> calls 42..47. The pass is read from the `cache_offset` the reference passes to every loop layer.
"""
import json
import math
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .compressed_experts import CompressedExperts, SafetensorsReader

WEIGHTS = "model-00001-of-00001.safetensors"
# short name -> (attribute path under layers[i].self_attn, tensor name under model.llm.layers.<i>.attn. in the file)
TARGETS = {
    "q": ("q_proj.linear", "wq_du.weight"),
    "k": ("k_proj.linear", "wk_dv.weight"),
    "v": ("v_proj.linear", "wv_dv.weight"),
    "o": ("o_proj.linear", "wo_ud.weight"),
}
CHUNK_MIN = 2048        # examples longer than this use the chunked attention
ATTN_CHUNK = 1024       # query rows per attention block
HEAD_CHUNK = 1024       # trained positions per head block (fp32 logits: 0.8 MB a row)


# ----------------------------------------------------------------------------------------------------- call identity
class CallTracker:
    """Which of the 48 layer evaluations is running. Set by a pre-hook on every layer's self_attn (a submodule, so the
    hook runs again when the checkpointed layer is recomputed; HF checkpoints the layer's forward, so hooks on the
    layer itself would not)."""

    def __init__(self, cfg):
        self.n_layers, self.ls, self.le = cfg.num_hidden_layers, cfg.loop_layer_index_start, cfg.loop_layer_index_end
        self.passes = cfg.loop_num_passes
        self.nloop = self.le - self.ls
        self.n_calls = self.n_layers + (self.passes - 1) * self.nloop
        self.cur = None
        self.picks = {}          # call -> top-k indices of the forward
        self.stored = self.replayed = 0

    def call_index(self, layer, cache_offset=0):
        if layer < self.ls:
            return layer
        if layer >= self.le:
            return layer + (self.passes - 1) * self.nloop
        p = 0 if not cache_offset else (cache_offset - self.n_layers + self.ls) // self.nloop + 1
        return self.ls + p * self.nloop + (layer - self.ls)

    def new_forward(self):
        self.picks.clear()
        self.stored = self.replayed = 0


class ReplayRouter(nn.Module):
    """Inkling2LoopTopKRouter's arithmetic, with the forward's picks replayed on the backward recompute."""

    def __init__(self, gate, tracker):
        super().__init__()
        self.gate = gate
        self.tracker = tracker

    def forward(self, hidden_states):
        g, tr = self.gate, self.tracker
        flat = hidden_states.float().reshape(-1, g.hidden_dim)
        logits = F.linear(flat, g.weight)
        scores = g.score_fn(logits)
        call = tr.cur
        if torch._C._current_graph_task_id() != -1:          # inside the backward: a checkpoint recompute
            if call not in tr.picks:
                raise RuntimeError(f"router replay: no stored picks for call {call}")
            idx = tr.picks[call]
            if idx.shape[0] != flat.shape[0]:
                raise RuntimeError(f"router replay: call {call} has {flat.shape[0]} tokens, stored {idx.shape[0]}")
            tr.replayed += 1
        else:
            with torch.no_grad():
                idx = torch.topk(scores + g.e_score_correction_bias, g.top_k, dim=-1, sorted=False).indices
            if torch.is_grad_enabled():
                tr.picks[call] = idx
                tr.stored += 1
        w = scores.gather(1, idx)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, w * g.routed_scaling_factor, idx


# -------------------------------------------------------------------------------------------------------- ablation
class EngineAblation:
    """ie-ablation-v1 as the engine reads it (inkling2_read_ablation) and applies it (i2_ablate_add)."""

    def __init__(self, path, hidden, n_calls, device):
        j = json.load(open(path))
        if j.get("format") != "ie-ablation-v1":
            raise ValueError(f"{os.path.basename(path)}: format is not ie-ablation-v1")
        if j.get("arch", "inkling2_loop") != "inkling2_loop" or int(j["hidden"]) != hidden:
            raise ValueError(f"{os.path.basename(path)}: not for this model")
        blob = j["file"] if j["file"].startswith("/") else os.path.join(os.path.dirname(path), j["file"])
        rows = int(j["rows"])
        dirs = torch.from_file(blob, size=rows * hidden, dtype=torch.float32).view(rows, hidden).clone()
        if not torch.isfinite(dirs).all():
            raise ValueError("ablation: non-finite direction value")
        self.path, self.name = path, j.get("name", "")
        self.emb = [None, None]
        self.attn = [[None, None] for _ in range(n_calls)]     # [remove (D, s, keep_norm), add v]
        self.moe = [[None, None] for _ in range(n_calls)]

        def calls(v):
            if v is None:
                return []
            if v == "all":
                return list(range(n_calls))
            return [int(x) for x in v]

        for g in j["groups"]:
            add = g.get("mode", "remove") == "add"
            r0, k = int(g["rows"][0]), int(g["rows"][1])
            s = float(g.get("strength", 1.0))
            D = dirs[r0:r0 + k]
            if add:
                if s == 0.0:
                    continue
                item = (s * D.double().sum(0)).float().to(device)
            else:
                G = D.double() @ D.double().T
                if (G - torch.eye(k, dtype=torch.float64)).abs().max() > 1e-3:
                    raise ValueError("ablation: a group's rows are not orthonormal")
                item = (D.to(device), s, bool(g.get("keep_norm", False)))
            slot = 1 if add else 0
            if g.get("embed"):
                self.emb[slot] = item
            for i in calls(g.get("attn")):
                self.attn[i][slot] = item
            for i in calls(g.get("moe")):
                self.moe[i][slot] = item
        n = lambda t, i: sum(1 for x in t if x[i] is not None)
        self.summary = (f"{os.path.basename(path)} ({self.name}): {n(self.attn, 0)} attention / {n(self.moe, 0)} MoE "
                        f"writes ablated, {n(self.attn, 1)} / {n(self.moe, 1)} steered, embedding "
                        f"{'yes' if any(x is not None for x in self.emb) else 'no'}")

    @staticmethod
    def apply(x, spec):
        rem, add = spec
        if rem is None and add is None:
            return x
        y = x.float()
        if rem is not None:
            D, s, keep_norm = rem
            c = y @ D.T                                        # (w . r_j)
            out = y - s * (c @ D)
            if keep_norm:
                ss = (y * y).sum(-1, keepdim=True)
                dd = (c * c).sum(-1, keepdim=True)
                s2 = 2 * s - s * s
                out = out * torch.sqrt(ss / torch.maximum(ss - s2 * dd, 1e-12 * ss + 1e-30))
            y = out
        if add is not None:
            y = y + add
        return y.to(x.dtype)


# ----------------------------------------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """Frozen bias-free nn.Linear + rank-r update. The forward uses W' = bf16(fp32(W) + (alpha / r) * B @ A),
    materialised once per model forward (the export writes exactly this tensor). With B = 0, bit-identical to W."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        assert base.bias is None
        self.base = base
        self.r, self.scale = r, alpha / r
        dev = base.weight.device
        A = torch.empty(r, base.in_features, dtype=torch.float32)        # drawn on the CPU generator: the same
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))                      # adapter on every card and on the CPU
        self.lora_A = nn.Parameter(A.to(dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=dev))
        self._w = None

    def merged_weight(self):
        return (self.base.weight.float() + self.scale * (self.lora_B @ self.lora_A)).to(self.base.weight.dtype)

    def forward(self, x):
        return F.linear(x, self._w)


def add_lora(model, r=16, alpha=32.0, targets=("q", "k", "v", "o"), seed=0):
    torch.manual_seed(seed)
    out = {}
    L = model.model.language_model.layers
    for i in range(len(L)):
        for t in targets:
            path = TARGETS[t][0].split(".")
            parent = L[i].self_attn
            for a in path[:-1]:
                parent = getattr(parent, a)
            base = getattr(parent, path[-1])
            w = LoRALinear(base, r, alpha)
            setattr(parent, path[-1], w)
            out[f"{i}.{t}"] = w

    def materialise(*_args, **_kw):                  # once per forward, outside the checkpointed layers
        for w in out.values():
            w._w = w.merged_weight()
    model.model.language_model.register_forward_pre_hook(materialise)    # the reference path (lm(...))
    model._ie_materialise = materialise                                   # lm_forward below
    return out


# ----------------------------------------------------------------------------------------------- chunked attention
def _repeat_kv(x, n):
    if n == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n, t, d).reshape(b, h * n, t, d)


def _attn_block(mod, q, k, v, rs, sinks, s, e, ks):
    """Rows s..e-1 against keys ks..e-1: the reference's eager_attention_forward arithmetic for those rows."""
    rpb = mod.relative_position_bias
    k = _repeat_kv(k, mod.num_key_value_groups)
    v = _repeat_kv(v, mod.num_key_value_groups)
    w = torch.matmul(q, k.transpose(2, 3)) * mod.scaling
    rel = (rs @ rpb.proj).transpose(1, 2)                                 # [b, heads, rows, rel_extent]
    dev = q.device
    dist = torch.arange(s, e, device=dev)[:, None] - torch.arange(ks, e, device=dev)[None, :]
    gi = dist.clamp(0, rpb.rel_extent - 1).expand(*rel.shape[:2], -1, -1)
    bias = rel.gather(-1, gi).masked_fill((dist < 0) | (dist >= rpb.rel_extent), 0.0)
    w = w + bias
    allowed = dist >= 0
    if mod.sliding_window:
        allowed = allowed & (dist < mod.sliding_window)
    w = w + torch.zeros(dist.shape, dtype=w.dtype, device=dev).masked_fill(~allowed, torch.finfo(w.dtype).min)
    p = F.softmax(torch.cat([w, sinks], dim=-1).float(), dim=-1)[..., :-1].to(q.dtype)
    return torch.matmul(p, v)


def _chunked_forward(self, hidden_states, key_hidden_states=None, value_hidden_states=None, attention_mask=None,
                     relative_distance=None, conv_mask=None, cache=None, cache_offset=0, **kwargs):
    T = hidden_states.shape[1]
    if T <= CHUNK_MIN or cache is not None:
        return self._ref_forward(hidden_states=hidden_states, key_hidden_states=key_hidden_states,
                                 value_hidden_states=value_hidden_states, attention_mask=attention_mask,
                                 relative_distance=relative_distance, conv_mask=conv_mask, cache=cache,
                                 cache_offset=cache_offset, **kwargs)
    pos = kwargs["position_ids"]
    khs = hidden_states if key_hidden_states is None else key_hidden_states
    vhs = hidden_states if value_hidden_states is None else value_hidden_states
    q = self.q_proj(hidden_states, pos)
    k = self.k_proj(khs, cache=None, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs)
    v = self.v_proj(vhs, cache=None, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs)
    rpb = self.relative_position_bias
    rs = rpb.norm(rpb.linear(hidden_states).view(*hidden_states.shape[:-1], rpb.num_heads, -1), pos)
    sinks = self.sink_proj(hidden_states).transpose(1, 2).unsqueeze(-1)
    outs = []
    rows = max(128, min(ATTN_CHUNK, (1 << 22) // T))           # rows x keys <= 4M: ~400 MB per fp32 score tensor
    for s in range(0, T, rows):
        e = min(T, s + rows)
        ks = max(0, s - self.sliding_window + 1) if self.sliding_window else 0
        outs.append(checkpoint(_attn_block, self, q[:, :, s:e], k[:, :, ks:e], v[:, :, ks:e], rs[:, s:e],
                               sinks[:, :, s:e], s, e, ks, use_reentrant=False))
    out = torch.cat(outs, dim=2).transpose(1, 2).contiguous().reshape(*hidden_states.shape[:-1], -1).contiguous()
    return self.o_proj(out, hidden_states), None


# ------------------------------------------------------------------------------------------------------------ load
def load(model_dir, device, threads=12, ablation=True, log=print):
    """The reference load (experts decoded to bf16 in RAM, ~63 GiB peak), then the experts are swapped for their
    compressed form and the model moves to `device`. Returns (tokenizer, model, tracker, ablation or None)."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lm = model.model.language_model
    reader = SafetensorsReader(os.path.join(model_dir, WEIGHTS))
    for i, layer in enumerate(lm.layers):
        old = layer.mlp.experts
        layer.mlp.experts = CompressedExperts(reader, i, old.act_fn, amax="exact")
        del old
    import gc
    gc.collect()
    # the vision tower is not trained and no training example carries an image: it stays on the CPU
    on_card = [lm, model.lm_head, model.logit_softcap]
    emb_w = lm.embed_tokens.weight.data            # the input embedding table stays in host memory (0.93 GiB):
    for m in on_card:                              # a forward only looks up T rows of it
        m.to(device)
    if str(device) != "cpu":
        lm.embed_tokens.weight = nn.Parameter(emb_w, requires_grad=False)

        def emb_forward(self, input_ids):
            rows = F.embedding(input_ids.to("cpu"), self.weight, self.padding_idx)
            return self.embed_norm(rows.to(self.embed_norm.weight.device))
        lm.embed_tokens.forward = types.MethodType(emb_forward, lm.embed_tokens)
    tracker = CallTracker(lm.config)
    for layer in lm.layers:
        att = layer.self_attn
        att.register_forward_pre_hook(
            lambda mod, args, kwargs: setattr(tracker, "cur", tracker.call_index(mod.layer_idx, kwargs.get("cache_offset", 0))),
            with_kwargs=True)
        att._ref_forward = att.forward
        att.forward = types.MethodType(_chunked_forward, att)
        layer.mlp.gate = ReplayRouter(layer.mlp.gate, tracker)
    abl = None
    ap = os.path.join(model_dir, "ie_ablation.json")
    if ablation and os.path.exists(ap):
        abl = EngineAblation(ap, lm.config.hidden_size, tracker.n_calls, device)
        lm.embed_tokens.register_forward_hook(lambda mod, args, out: EngineAblation.apply(out, abl.emb))
        for layer in lm.layers:
            for conv, table in ((layer.post_attention_convolution, abl.attn), (layer.post_mlp_convolution, abl.moe)):
                conv.register_forward_hook(
                    lambda mod, args, kwargs, out, table=table:
                        EngineAblation.apply(out, table[tracker.call_index(mod.layer_idx, kwargs.get("cache_offset", 0))]),
                    with_kwargs=True)
        log(f"run-time ablation applied in the training forward: {abl.summary}")
    return tok, model, tracker, abl


def enable_grad_ckpt(model):
    """HF's per-layer checkpointing, for the reference loop only (lm_forward checkpoints by itself)."""
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()                    # train() only switches the checkpointing on: the model has no dropout


# ------------------------------------------------------------------------------------------------- the layer loop
def lm_forward(model, input_ids, ckpt=True):
    """Inkling2LoopTextModel.forward for one sequence, no cache, the same calls in the same order (modeling file
    1308-1447), with every layer CALL -- the messages it reads (passes 2 and 3) and the layer -- under one
    checkpoint, and the final read-out under another. The reference checkpoints only the layer, so the 24 message
    tensors ([T, 4, hidden] fp32) and the message / read-out attention graphs stayed alive for the whole backward
    (about 7 GiB at 8,192 tokens). Masks and [T, T] distance tables are built only when the reference attention runs
    (T <= CHUNK_MIN); the chunked attention computes its own per block."""
    import transformers.models.inkling2_loop.modeling_inkling2_loop as MM
    lm = model.model.language_model
    cfg = lm.config
    if getattr(model, "_ie_materialise", None):
        model._ie_materialise()
    x = lm.embed_tokens(input_ids)
    T = x.shape[1]
    pos = torch.arange(T, device=x.device).unsqueeze(0)
    mk = {"config": cfg, "inputs_embeds": x, "attention_mask": None, "cache": None, "position_ids": pos}
    conv_mask = MM.create_recurrent_attention_mask(**mk)
    if T <= CHUNK_MIN:
        masks = {"full_attention": MM.create_causal_mask(**mk), "sliding_attention": MM.create_sliding_window_causal_mask(**mk)}
        dist = {"full_attention": MM.get_relative_distance(x, None, is_sliding=False),
                "sliding_attention": MM.get_relative_distance(x, None, is_sliding=True)}
    else:
        masks = dist = {"full_attention": None, "sliding_attention": None}
    use = ckpt and torch.is_grad_enabled()

    def run(layer, h, bs, prev_bs=None, prev_h=None, li=0, cache_offset=None):
        kw = dict(attention_mask=masks[layer.attention_type], relative_distance=dist[layer.attention_type],
                  conv_mask=conv_mask, cache=None, position_ids=pos)
        if cache_offset is not None:
            kw["cache_offset"] = cache_offset

        def f(h, bs, prev_bs, prev_h):
            if prev_bs is None:
                return layer(h, bs, **kw)
            return layer(h, bs, lm.message_input(prev_bs, prev_h, li), **kw)
        return checkpoint(f, h, bs, prev_bs, prev_h, use_reentrant=False) if use else f(h, bs, prev_bs, prev_h)

    ls, le = cfg.loop_layer_index_start, cfg.loop_layer_index_end
    h = x
    bs = x.new_zeros(*x.shape[:-1], 0, x.shape[-1])

    def step(layer, h, bs, **k):
        nh = run(layer, h, bs, **k)
        if layer.is_block_boundary:
            bs = torch.cat([bs, h.unsqueeze(-2)], dim=-2)
        return nh, bs

    for layer in lm.layers[:ls]:
        h, bs = step(layer, h, bs)
    pre_h, pre_bs = h, bs
    npre = pre_bs.shape[-2]
    for layer in lm.layers[ls:le]:
        h, bs = step(layer, h, bs)
    for p in range(1, cfg.loop_num_passes):
        prev_bs, prev_h = bs[..., npre + 1:, :], h
        h, bs = pre_h, pre_bs
        off = cfg.num_hidden_layers + (p - 1) * (le - ls) - ls
        for li, layer in enumerate(lm.layers[ls:le]):
            h, bs = step(layer, h, bs, prev_bs=prev_bs, prev_h=prev_h, li=li, cache_offset=off)
    for layer in lm.layers[le:]:
        h, bs = step(layer, h, bs)

    def final(bs, h):
        return lm.norm(lm.final_input(bs, h)[0]).to(x.dtype)
    return checkpoint(final, bs, h, use_reentrant=False) if use else final(bs, h)


# ------------------------------------------------------------------------------------------------------------ loss
def _head_block(model, h, y):
    logits = model.logit_softcap(model.lm_head(h).float())
    return F.cross_entropy(logits, y, reduction="sum"), (logits.argmax(-1) == y).sum()


def example_loss(model, tracker, ids, labels, device, reference=False, ckpt=True):
    """Mean cross-entropy over the trained positions of one example (labels != -100), and the number of trained
    positions where the greedy next token is right. ids / labels: python lists, labels aligned with ids."""
    tracker.new_forward()
    x = torch.tensor([ids], device=device)
    tgt = torch.tensor(labels[1:], device=device)
    sel = (tgt != -100).nonzero(as_tuple=True)[0]
    if reference:                 # the publisher's loop (with HF's per-layer checkpointing if enabled)
        h = model.model.language_model(input_ids=x, use_cache=False).last_hidden_state[0, :-1]
    else:
        h = lm_forward(model, x, ckpt=ckpt)[0, :-1]
    h, y = h[sel], tgt[sel]
    total, correct = 0.0, 0
    for s in range(0, y.numel(), HEAD_CHUNK):
        if torch.is_grad_enabled():
            l, c = checkpoint(_head_block, model, h[s:s + HEAD_CHUNK], y[s:s + HEAD_CHUNK], use_reentrant=False)
        else:
            l, c = _head_block(model, h[s:s + HEAD_CHUNK], y[s:s + HEAD_CHUNK])
        total = total + l
        correct += int(c)
    return total / max(1, y.numel()), correct, int(y.numel())


def save_adapter(adapters, path, meta):
    from safetensors.torch import save_file
    w0 = next(iter(adapters.values()))
    md = dict(meta, r=w0.r, scale=w0.scale, mode="merged", keys=sorted(adapters))
    t = {}
    for k, w in adapters.items():
        t[k + ".A"] = w.lora_A.detach().cpu().contiguous()
        t[k + ".B"] = w.lora_B.detach().cpu().contiguous()
    save_file(t, path, metadata={"inkling2_lora": json.dumps(md)})
