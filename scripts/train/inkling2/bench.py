"""Checks and measurements of the card training path (milestone 1).

  xpu_env.sh venv/bin/python -I bench.py card  --model DIR --device xpu:1 --out DIR [--lengths 1024,4096,8192]
  xpu_env.sh venv/bin/python -I bench.py cpu   --model DIR --out DIR          (after `card`: the same gradient on the CPU)

card: (1) gradient of one 512-token example without gradient checkpointing, then with per-layer checkpointing and the
router replay (must agree, and every router call of the forward must be replayed in the backward); (2) the chunked
attention against the reference attention at 3,000 tokens (loss and trained-position logits); (3) seconds, tokens per
second and peak card memory of one training step (forward + backward of one example) at each length.
Example text: a Dream session from the OpenAI-shaped twin file rendered with the model's chat template, cut to length;
the last 25 % of the positions are trained. Prints numbers only, never example text.
"""
import argparse, json, os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from inkling2 import model as M

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["card", "cpu"])
ap.add_argument("--model", required=True)
ap.add_argument("--device", default="xpu:1")
ap.add_argument("--out", required=True)
ap.add_argument("--lengths", default="1024,4096,8192,12288,16384")
ap.add_argument("--text", default=os.path.join(os.environ.get("IE_DATA_DRIVE", os.path.expanduser("~/data")), "dream-dataset/sft/dream_local_train.openai.jsonl"))
ap.add_argument("--no-ablation", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = a.device if a.mode == "card" else "cpu"


def card_used(card):
    s = subprocess.run(["xpu-smi", "stats", "-d", str(card)], capture_output=True, text=True).stdout
    m = re.search(r"GPU Memory Used \(MiB\).*?current:\s*(\d+)", s)
    return int(m.group(1)) if m else -1


t0 = time.time()
tok, model, tracker, abl = M.load(a.model, dev, ablation=not a.no_ablation)
print(f"loaded on {dev} in {time.time() - t0:.0f} s", flush=True)
if a.mode == "card":
    card = int(dev.split(":")[1])
    print(f"card {card}: torch allocated {torch.xpu.memory_allocated(dev) / 2**30:.2f} GiB, xpu-smi {card_used(card)} MiB", flush=True)

# a long token stream from real sessions (concatenated renders), cut per length
stream = []
for line in open(a.text):
    r = json.loads(line)
    msgs = []
    for m in r["messages"]:
        m = dict(m)
        for tc in m.get("tool_calls") or []:
            if isinstance(tc["function"].get("arguments"), str):
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"] or "{}")
        msgs.append(m)
    stream += tok.apply_chat_template(msgs, tools=r.get("tools"), tokenize=True, return_dict=False)
    if len(stream) > 17000:
        break


def example(T):
    ids = list(stream[:T])
    cut = int(T * 0.75)
    return ids, [-100] * cut + ids[cut:]


adapters = M.add_lora(model, seed=0)
params = [p for w in adapters.values() for p in (w.lora_A, w.lora_B)]
g = torch.Generator().manual_seed(1)
with torch.no_grad():                     # B != 0 so that A gets a gradient too
    for w in adapters.values():
        w.lora_B.copy_(torch.randn(w.lora_B.shape, generator=g) * 1e-3)


def grad_of(ids, labels, reference=False, ckpt=True):
    for p in params:
        p.grad = None
    loss, corr, n = M.example_loss(model, tracker, ids, labels, dev, reference=reference, ckpt=ckpt)
    loss.backward()
    return loss.item(), torch.cat([p.grad.detach().float().flatten().cpu() for p in params])


cos = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=0).item()
res = {"model": a.model, "device": dev, "ablation": abl.summary if abl else None}
ids, labels = example(256)
if a.mode == "cpu":
    l, gr = grad_of(ids, labels, reference=True)
    ref = torch.load(os.path.join(a.out, "grad_card.pt"))
    res["cpu_reference_loss"] = l
    for k in ("ref", "own_nockpt", "own_ckpt"):
        res[f"cpu_vs_card_{k}"] = {"cos": cos(gr, ref[k]), "norm_ratio": (ref[k].norm() / gr.norm()).item(),
                                  "loss_card": ref["loss_" + k]}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(a.out, "cpu.json"), "w"), indent=1)
    sys.exit(0)

# (1) gradient at 256 tokens: the reference loop (no checkpointing), the own loop without and with per-call
#     checkpointing + router replay (twice): with the deterministic expert combine these must agree exactly
lr_, gr = grad_of(ids, labels, reference=True)
ln, gn = grad_of(ids, labels, ckpt=False)
lc, gc_ = grad_of(ids, labels)
st, rp = tracker.stored, tracker.replayed
lc2, gc2 = grad_of(ids, labels)
res["grad_check_256"] = {"loss_ref": lr_, "loss_own_nockpt": ln, "loss_own_ckpt": lc, "loss_own_ckpt_again": lc2,
                         "router_calls_stored": st, "router_calls_replayed": rp,
                         "identical_own_nockpt_vs_ckpt": bool(torch.equal(gn, gc_)),
                         "identical_ckpt_runs": bool(torch.equal(gc_, gc2)),
                         "identical_ref_vs_own_nockpt": bool(torch.equal(gr, gn)),
                         "cos_ref_vs_own_ckpt": cos(gr, gc_), "max_abs_diff_nockpt_vs_ckpt": (gn - gc_).abs().max().item(),
                         "grad_norm": gn.norm().item()}
torch.save({"ref": gr, "own_nockpt": gn, "own_ckpt": gc_, "loss_ref": lr_, "loss_own_nockpt": ln, "loss_own_ckpt": lc},
           os.path.join(a.out, "grad_card.pt"))
print("grad check:", json.dumps(res["grad_check_256"]), flush=True)

# (2) forwards without gradient: own loop vs reference loop at 512 (same calls: should agree to the card's noise),
#     and at 3,000 tokens (own loop with the chunked attention vs the reference loop with the reference attention)
with torch.no_grad():
    out = {}
    for T in (256, 2600):
        ids3, lab3 = example(T)
        lo, co, n3 = M.example_loss(model, tracker, ids3, lab3, dev)
        lr_, cr, _ = M.example_loss(model, tracker, ids3, lab3, dev, reference=True)
        out[T] = {"loss_own": lo.item(), "loss_reference": lr_.item(), "greedy_correct_own": co,
                  "greedy_correct_reference": cr, "positions": n3}
res["forward_check"] = out
print("forward check:", json.dumps(out), flush=True)

# (3) speed and memory of one training step per length
res["speed"] = []
for T in [int(x) for x in a.lengths.split(",") if x]:
    ids, labels = example(T)
    times = []
    import gc; gc.collect()
    torch.xpu.empty_cache()
    torch.xpu.reset_peak_memory_stats(dev)
    for rep in range(2 if T > 8192 else 3):
        torch.xpu.synchronize(dev)
        ts = time.time()
        try:
            grad_of(ids, labels)
        except torch.OutOfMemoryError:
            print(f"  T={T}: out of card memory", flush=True)
            res["speed"].append({"tokens": T, "out_of_memory": True})
            times = None
            break
        torch.xpu.synchronize(dev)
        times.append(time.time() - ts)
        print(f"  T={T} rep {rep}: {times[-1]:.1f} s", flush=True)
    if times is None:
        torch.xpu.empty_cache()
        break
    sec = sum(times[1:]) / len(times[1:])
    r = {"tokens": T, "seconds_per_example": sec, "first_step_seconds": times[0], "tokens_per_second": T / sec,
         "peak_allocated_gib": torch.xpu.max_memory_allocated(dev) / 2**30,
         "peak_reserved_gib": torch.xpu.max_memory_reserved(dev) / 2**30, "xpu_smi_mib_after": card_used(card)}
    res["speed"].append(r)
    print("speed:", json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(a.out, "card.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
