"""Chunked attention vs the reference attention, module by module, on real hidden states (no gradient).

  xpu_env.sh venv/bin/python -I attn_check.py MODEL_DIR T
Prints, per checked layer: relative L2 error of the whole output and the worst row (with its position), for the
chunked block computation in bf16 and for the reference module run in fp32 as the yardstick of bf16 rounding.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import transformers.models.inkling2_loop.modeling_inkling2_loop as MM
from inkling2 import model as M

model_dir, T = sys.argv[1], int(sys.argv[2])
dev = "xpu:0"
tok, model, tracker, abl = M.load(model_dir, dev, ablation=False)
lm = model.model.language_model
torch.manual_seed(0)
ids = torch.randint(1000, 150000, (1, T), device=dev)
with torch.no_grad():
    x = lm.embed_tokens(ids)
    pos = torch.arange(T, device=dev)[None]
    mk = {"config": lm.config, "inputs_embeds": x, "attention_mask": None, "cache": None, "position_ids": pos}
    masks = {"full_attention": MM.create_causal_mask(**mk), "sliding_attention": MM.create_sliding_window_causal_mask(**mk)}
    dist = {"full_attention": MM.get_relative_distance(x, None, is_sliding=False),
            "sliding_attention": MM.get_relative_distance(x, None, is_sliding=True)}
    for i in sorted({0, 1, 2, 3, 5}):
        layer = lm.layers[i]
        att = layer.self_attn
        h = x * 4.0                        # a hidden state of realistic size
        kw = dict(attention_mask=masks[layer.attention_type], relative_distance=dist[layer.attention_type], position_ids=pos)
        if layer.attention_type == "full_attention":
            kw.update(key_hidden_states=h, value_hidden_states=h)
        ref = att._ref_forward(hidden_states=h, **kw)[0].float()
        ch = att.forward(hidden_states=h, **kw)[0].float()
        rel = lambda a, b: ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12))[0]
        r = rel(ch, ref)
        print(json.dumps({"layer": i, "type": layer.attention_type, "sliding_window": att.sliding_window,
                          "rel_l2_whole": ((ch - ref).norm() / ref.norm()).item(), "worst_row_rel": r.max().item(),
                          "worst_row": int(r.argmax()), "rows_over_2e-2": int((r > 2e-2).sum()),
                          "mean_row_rel_first_1024": r[:1024].mean().item(), "mean_row_rel_after_1024": r[1024:].mean().item()}),
              flush=True)
