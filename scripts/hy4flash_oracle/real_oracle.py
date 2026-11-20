#!/usr/bin/env python3
"""Hy4-flash REAL-weights oracle for the first K layers (docs/hy4flash/00_PORT_AND_ABLITERATION_PLAN.md, phase 3).

  python real_oracle.py --source <Tencent dir> --layers 4 --ids-file ids.txt --out <dir>

The mini gate proves the engine's arithmetic on random weights. It cannot see a fault that only real weights excite.
This runs the reference implementation itself on the released weights -- the first K layers only (fp32 on the CPU; the
n-gram table stays BF16 and is cast after the lookup, as the reference does) -- on the given token ids and writes

  <out>/oracle/ids.i32, streams.f32 ([K + 1, T, hc, dim]: the streams entering layers 0 .. K-1, then leaving layer K-1),
  routes.i32 / route_margin.f32 ([MoE layers among the K, T, top_k] / [.., T]), meta.json

for `ie-hy4flash-mini <engine dir> --oracle <out>/oracle --first-layers K --exact-experts`.

With --layers equal to the model's layer count (42) it is the FULL model: logits.f32 ([T, vocab]) is written too and the
streams file holds the 42 layer entries only (`ie-hy4flash-mini <engine dir> --oracle <out>/oracle --exact-experts`).

Memory: the routed experts and the n-gram tables are never copied -- they are read-only mappings of the shards, cast to
fp32 one expert (or one looked-up row) at a time -- so what is held is the dense part as fp32 (about 25 GB for the full
model) plus page cache. A full forward costs about 15 ms per (layer, expert) used. Run with the reference venv.
"""
import argparse, glob, json, os, struct, sys, time, warnings
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy4flash_common import release_name  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers.initialization import no_init_weights  # noqa: E402
from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashTextConfig  # noqa: E402
from transformers.models.hy_v4_flash.modeling_hy_v4_flash import HYV4FlashExperts, HYV4FlashTextModel  # noqa: E402

def lazy_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """HYV4FlashExperts.forward with the expert's two matrices cast to the activations' dtype as they are used (they
    stay read-only BF16 mappings of the release)."""
    final = torch.zeros_like(hidden_states)
    mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in hit:
        e = expert_idx[0]
        if e == self.num_experts: continue
        top_k_pos, token_idx = torch.where(mask[e])
        current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[e].to(hidden_states.dtype)))
        current = F.linear(current, self.down_proj[e].to(hidden_states.dtype)) * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, current.to(final.dtype))
    return final

def load(source, layers, bf16=False):
    """The reference model on the released weights: the first `layers` layers (None = all), experts and n-gram tables mapped in place."""
    K = layers if layers else json.load(open(os.path.join(source, "config.json")))["text_config"]["num_hidden_layers"]
    tc = dict(json.load(open(os.path.join(source, "config.json")))["text_config"])
    for k in ("model_type", "transformers_version", "architectures"): tc.pop(k, None)
    tc["num_hidden_layers"] = K
    tc["layer_types"] = tc["layer_types"][:K]; tc["mlp_layer_types"] = tc["mlp_layer_types"][:K]
    ids_e = [i for i in (tc.get("engram_layer_ids") or []) if i <= K]
    tc["engram_layer_ids"] = ids_e or None
    tc["engram_layer_head_vocab_sizes"] = {str(i): tc["engram_layer_head_vocab_sizes"][str(i)] for i in ids_e} or None
    cfg = HYV4FlashTextConfig(**tc)
    idx, where = {}, {}
    for f in sorted(glob.glob(os.path.join(source, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__": idx[k] = f; where[k] = (f, 8 + n + v["data_offsets"][0], v["shape"], v["dtype"])
    def lazy_bf16(name):                 # a read-only mapping of the tensor's bytes in its shard, viewed as BF16
        f, off, shape, dt = where[name]
        assert dt == "BF16", (name, dt)
        mm = np.memmap(f, dtype=np.uint16, mode="r", offset=off, shape=tuple(shape))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.from_numpy(mm).view(torch.bfloat16)
    n_total = json.load(open(os.path.join(source, "config.json")))["text_config"]["num_hidden_layers"]
    full = K == n_total
    HYV4FlashExperts.forward = lazy_experts_forward
    t0 = time.time()
    torch.set_default_dtype(torch.bfloat16)                 # the big tensors are born BF16 (half the memory), then cast one by one
    with no_init_weights(): model = HYV4FlashTextModel(cfg)
    torch.set_default_dtype(torch.float32)
    model.eval()
    handles = {}
    def get(name):
        f = idx[name]
        if f not in handles: handles[f] = safe_open(f, framework="pt")
        return handles[f].get_tensor(name)
    n_loaded = 0
    with torch.no_grad():
        for name, p in list(model.named_parameters()) + [(n, b) for n, b in model.named_buffers() if n in model.state_dict()]:
            rn = release_name(name)
            lazy = "multi_head_embedding" in name or name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj")
            src = lazy_bf16(rn) if lazy else get(rn)
            assert tuple(src.shape) == tuple(p.shape), (name, src.shape, p.shape)
            p.data = src if (lazy or bf16 or not torch.is_floating_point(src)) else src.float()
            n_loaded += 1
    for mod in model.modules():                              # anything floating that is still BF16 (derived buffers) -> fp32
        for bn, b in list(mod._buffers.items()):
            if b is not None and b.dtype == torch.bfloat16 and not bf16: mod._buffers[bn] = b.float()
    n_eps = 0
    for mod in model.modules():                              # the released model runs in bf16: nn.RMSNorm's default eps there
        if isinstance(mod, torch.nn.RMSNorm) and mod.eps is None: mod.eps = 2.0 ** -7; n_eps += 1
    print(f"first {K} layers of the release loaded: {n_loaded} tensors, {time.time() - t0:.0f} s; n-gram layers among them: {ids_e}", flush=True)
    return model, cfg, full, get

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ids-file", required=True); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--stats", action="store_true", help="also record what the low-bit ideas depend on (see the end of main)")
    ap.add_argument("--bf16", action="store_true", help="run in bfloat16, the dtype the release is served in (slow on a CPU): how far the vendor's own arithmetic sits from the fp32 run")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    K = a.layers
    model, cfg, full, get = load(a.source, K, a.bf16)
    ids = torch.tensor([[int(x) for x in open(a.ids_file).read().split()]], dtype=torch.long)
    T = ids.shape[1]
    streams, routes, margins = [], [], []
    hooks = [layer.register_forward_pre_hook(lambda m, args: streams.append(args[0].detach().float().clone())) for layer in model.layers]
    if not full: hooks.append(model.layers[-1].register_forward_hook(lambda m, args, out: streams.append(out.detach().float().clone())))
    def route_hook(m, args, out):
        logits, _, pick = out
        choice = logits.sigmoid() + m.e_score_correction_bias
        top = choice.topk(m.top_k + 1, dim=-1).values
        routes.append(pick.sort(dim=-1).values.int().clone()); margins.append((top[:, m.top_k - 1] - top[:, m.top_k]).float().clone())
    for layer in model.layers:
        if hasattr(layer.mlp, "gate"): hooks.append(layer.mlp.gate.register_forward_hook(route_hook))
    # --stats: per MoE layer, the second moment of the experts' latent INPUTS (x^T x, [latent, latent]) and of their
    # summed latent OUTPUTS, each row's 16 routing weights sorted, and every expert's pick count and weight mass
    st = {"xtx": [], "yty": [], "wsorted": [], "count": [], "mass": []}
    if a.stats:
        def experts_pre(m, args):
            x, pick, w = args[0].double(), args[1], args[2].double()
            st["xtx"].append((x.T @ x).float()); st["wsorted"].append(w.sort(dim=-1, descending=True).values.mean(0).float())
            c = torch.zeros(m.num_experts, dtype=torch.float64); ms = torch.zeros(m.num_experts, dtype=torch.float64)
            c.index_add_(0, pick.flatten(), torch.ones(pick.numel(), dtype=torch.float64)); ms.index_add_(0, pick.flatten(), w.flatten())
            st["count"].append(c.float()); st["mass"].append(ms.float())
        def experts_post(m, args, out):
            y = out.double(); st["yty"].append((y.T @ y).float())
        for layer in model.layers:
            if hasattr(layer.mlp, "experts"):
                hooks.append(layer.mlp.experts.register_forward_pre_hook(experts_pre)); hooks.append(layer.mlp.experts.register_forward_hook(experts_post))
    t0 = time.time()
    with torch.no_grad():
        ngram = model.build_ngram_tokens(ids, None) if cfg.engram_layer_ids else None
        mask = torch.ones_like(ids, dtype=torch.bool) if cfg.engram_layer_ids else None
        out = model(input_ids=ids, ngram_tokens=ngram, engram_mask=mask, use_cache=False)
        logits = F.linear(out.last_hidden_state.float(), get("lm_head.weight").float())[0] if full else None
    o = os.path.join(a.out, "oracle"); os.makedirs(o, exist_ok=True)
    ids[0].numpy().astype(np.int32).tofile(os.path.join(o, "ids.i32"))
    np.stack([s[0].numpy() for s in streams]).astype(np.float32).tofile(os.path.join(o, "streams.f32"))
    if full:
        logits.numpy().astype(np.float32).tofile(os.path.join(o, "logits.f32"))
        np.zeros((0,), np.int32).tofile(os.path.join(o, "recall.i32"))
        top = torch.topk(torch.log_softmax(logits[-1].double(), -1), 5)
        print("last position, top 5 next tokens (id: probability):", ", ".join(f"{int(i)}: {100 * float(v.exp()):.1f} %" for v, i in zip(top.values, top.indices)))
    np.stack([r.numpy() for r in routes]).astype(np.int32).tofile(os.path.join(o, "routes.i32"))
    np.stack([m.numpy() for m in margins]).astype(np.float32).tofile(os.path.join(o, "route_margin.f32"))
    json.dump({"tokens": T, "first_layers": K, "full": full, "streams": len(streams), "hc": cfg.hc_mult, "hidden": cfg.hidden_size, "top_k": cfg.num_experts_per_tok,
               "moe_layers": len(routes), "source": a.source}, open(os.path.join(o, "meta.json"), "w"), indent=1)
    if a.stats:
        np.savez(os.path.join(o, "stats.npz"), tokens=T, **{k: torch.stack(v).numpy() for k, v in st.items()})
        print("stats.npz written:", {k: tuple(torch.stack(v).shape) for k, v in st.items()})
    mg = torch.stack(margins)
    print(f"forward {time.time() - t0:.0f} s; T = {T}; stream rms entering each layer and at the end: " + " ".join(f"{float(s.pow(2).mean().sqrt()):.3f}" for s in streams))
    print(f"route margins: median {float(mg.median()):.4f}, {int((mg < 1e-3).sum())} of {mg.numel()} under 0.001")

if __name__ == "__main__": main()
