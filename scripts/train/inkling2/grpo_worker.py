"""One GRPO training process on one card (grpo.py starts one per card and drives them through files in the work folder).

The cards alternate between the engine (which writes the attempts) and these processes (which learn from them). Between
steps a worker PARKS: the model moves to host memory and the card is emptied for the engine; it un-parks for the next
step. Files (all in --work):
  parked<rank>.<k>      this worker is parked, ready for step k+1 (k = steps done)
  batch.<k>.json        the scored attempts of step k: [{"messages": [...], "tools": [...]|null, "adv": float}, ...]
  go.<k>                train step k now
  result.<k>.json       rank 0: {"loss", "trained_tokens", "seconds", "grad_norm"} of step k
  finish.json           {"step": n, "training_md": "..."}: rank 0 writes the model folder from the adapter of step n
  failed                some process failed
Rank 0 also keeps <work>/policy, the model folder the engine serves: a copy of the base whose 96 attention tensors are
rewritten from the adapter after every step, so the attempts always come from the current model.

GRPO step (beta = 0, one update per batch, as in DAPO / Dr. GRPO): loss = sum over attempts i and their reply tokens t
of adv_i * (-log p(t)) / (all reply tokens of the batch). With one update per batch the ratio is 1 and clipping never
acts, so the gradient is the plain policy gradient with group-normalised advantages.
"""
import argparse
import gc
import json
import math
import os
import shutil
import signal
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STOP = {"flag": False}
signal.signal(signal.SIGTERM, lambda *_: STOP.__setitem__("flag", True))
signal.signal(signal.SIGINT, lambda *_: STOP.__setitem__("flag", True))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--world", type=int, default=1)
ap.add_argument("--card", type=int, required=True)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--lora-rank", type=int, default=16)
ap.add_argument("--alpha", type=float, default=32.0)
ap.add_argument("--max-len", type=int, default=12288)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
R0 = a.rank == 0


def log(msg):
    print(f"[card {a.card}] {msg}", file=sys.stderr, flush=True)


def flag(name):
    return os.path.join(a.work, name)


def touch(name):
    open(flag(name) + ".tmp", "w").close()
    os.replace(flag(name) + ".tmp", flag(name))


def wait_any(names):
    """Block until one of the files exists; returns its name. Aborts when another process failed."""
    while True:
        for n in names:
            if os.path.exists(flag(n)):
                return n
        if os.path.exists(flag("failed")):
            raise SystemExit(1)
        time.sleep(0.5)


def make_policy_folder(base, dst):
    """A standalone copy of the base model folder for the engine (the weight file copied, the rest hard links)."""
    from inkling2.model import WEIGHTS
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst + ".writing")
    same_fs = os.stat(os.path.dirname(dst)).st_dev == os.stat(base).st_dev
    for name in sorted(os.listdir(base)):
        s, d = os.path.join(base, name), os.path.join(dst + ".writing", name)
        if name in (WEIGHTS, ".cache", "TRAINING.md", "EXPORT_INFO.json") or name.endswith(".writing"):
            continue
        if os.path.isdir(s):
            shutil.copytree(s, d, copy_function=os.link if same_fs else shutil.copy2)
        elif name.startswith("ie_") or name.endswith(".f32") or not same_fs:
            shutil.copy2(s, d)
        else:
            os.link(s, d)
    shutil.copyfile(os.path.join(base, WEIGHTS), os.path.join(dst + ".writing", WEIGHTS))
    os.rename(dst + ".writing", dst)


def main():
    import torch
    from inkling2 import data as D
    from inkling2 import model as M
    from inkling2.compressed_experts import SafetensorsReader
    from inkling2.export import export, patch_attention

    dev = "xpu:0"
    policy = flag("policy")
    if R0:
        t0 = time.time()
        make_policy_folder(a.model, policy)
        log(f"policy folder for the engine written in {time.time() - t0:.0f} s")
    if a.rank:
        while not os.path.exists(flag(f"loaded{a.rank - 1}")):
            if os.path.exists(flag("failed")):
                raise SystemExit(1)
            time.sleep(1)
    t0 = time.time()
    tok, model, tracker, abl = M.load(a.model, dev, log=log)
    touch(f"loaded{a.rank}")
    log(f"model on the card in {time.time() - t0:.0f} s")
    adapters = M.add_lora(model, r=a.lora_rank, alpha=a.alpha, seed=a.seed)
    keys = sorted(adapters)
    params = [p for k in keys for p in (adapters[k].lora_A, adapters[k].lora_B)]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    meta = {"scale": a.alpha / a.lora_rank, "keys": keys}
    lm = model.model.language_model
    on_card = [lm, model.lm_head, model.logit_softcap]
    emb_cpu = lm.embed_tokens.weight.data                 # the embedding table lives in host memory (see model.load)

    dist = None
    if a.world > 1:
        import torch.distributed as dist
        dist.init_process_group("gloo", init_method="file://" + flag("dist_init"), rank=a.rank, world_size=a.world)

    def allsum(values):
        t = torch.tensor(values, dtype=torch.float64)
        if dist:
            dist.all_reduce(t)
        return t.tolist()

    def park():
        for w in adapters.values():
            w._w = None
        for m in on_card:
            m.to("cpu")
        gc.collect()
        torch.xpu.synchronize()
        torch.xpu.empty_cache()

    def unpark():
        for m in on_card:
            m.to(dev)
        lm.embed_tokens.weight.data = emb_cpu
        gc.collect()
        torch.xpu.empty_cache()

    def adapter_tensors():
        return {**{k + ".A": adapters[k].lora_A.detach().cpu().clone() for k in keys},
                **{k + ".B": adapters[k].lora_B.detach().cpu().clone() for k in keys}}

    os.makedirs(flag("adapters"), exist_ok=True)
    rend = D.Renderer(tok)
    base_reader = SafetensorsReader(os.path.join(a.model, M.WEIGHTS)) if R0 else None

    def render(item):
        """-> [(ids, labels)]: a single-reply attempt trains its last turn; a whole session ("all_turns") trains every
        assistant turn of the model's own, cut into windows of --max-len by the supervised trainer's rule."""
        msgs, w = D.convert(item["messages"], item.get("tools"), 0)
        segs, _ = rend.segments(msgs, item.get("tools"))
        if item.get("all_turns"):
            w = [1 if m["role"] == "assistant" else 0 for m in msgs]
            return D.window(segs, w, msgs, a.max_len, {})
        w = [0] * (len(w) - 1) + [1]                          # only the attempt (the last turn) is trained
        ids, labels = [], []
        for s, wi in zip(segs, w):
            ids += s
            labels += s if wi else [-100] * len(s)
        return [(ids[:a.max_len], labels[:a.max_len])]

    park()
    k = 0
    touch(f"parked{a.rank}.0")
    while True:
        got = wait_any([f"go.{k + 1}", "finish.json"])
        if got == "finish.json":
            break
        k += 1
        ts = time.time()
        batch = json.load(open(flag(f"batch.{k}.json")))
        items, skipped = [], 0
        for it in batch:
            try:
                items += [{"adv": it["adv"], "ids": ids, "labels": labels} for ids, labels in render(it)]
            except Exception as e:                            # noqa: BLE001 -- one unrenderable session must not end a run
                skipped += 1
                log(f"an attempt could not be rendered and was skipped ({type(e).__name__}: {str(e)[:120]})")
        load, mine = [0] * a.world, []                        # longest first to the card with fewer tokens
        for i in sorted(range(len(items)), key=lambda i: -len(items[i]["ids"])):
            r = min(range(a.world), key=lambda r: (load[r], r))
            load[r] += len(items[i]["ids"])
            if r == a.rank:
                mine.append(items[i])
        n_all = sum(sum(1 for x in it["labels"][1:] if x != -100) for it in items)
        if not items:                                         # no group had a signal: nothing to learn this step
            if R0:
                torch.save(adapter_tensors(), flag(f"adapters/step{k}.pt"))
                with open(flag(f"result.{k}.json") + ".tmp", "w") as f:
                    json.dump({"loss": 0.0, "trained_tokens": 0, "seconds": 0.0, "grad_norm": 0.0,
                               "learning_rate": 0.0}, f)
                os.replace(flag(f"result.{k}.json") + ".tmp", flag(f"result.{k}.json"))
            touch(f"parked{a.rank}.{k}")
            continue
        unpark()
        lr = a.lr * min(1.0, k / max(1, a.warmup))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        lsum = 0.0
        for it in mine:
            try:
                ce, _, n = M.example_loss(model, tracker, it["ids"], it["labels"], dev)
                loss = it["adv"] * ce * n / max(1, n_all)
                loss.backward()
                lsum += loss.item()
            except torch.OutOfMemoryError:
                gc.collect()
                torch.xpu.empty_cache()
                log(f"an attempt of {len(it['ids'])} tokens did not fit on the card and was skipped")
        if dist:
            flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().cpu() for p in params])
            dist.all_reduce(flat)
            o = 0
            for p in params:
                p.grad = flat[o:o + p.numel()].view_as(p).to(p.device)
                o += p.numel()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        if not math.isfinite(gn):
            raise RuntimeError(f"the gradient became non-finite at step {k}")
        opt.step()
        opt.zero_grad(set_to_none=True)
        lsum = allsum([lsum])[0]
        park()
        if R0:
            ad = adapter_tensors()
            torch.save(ad, flag(f"adapters/step{k}.pt"))
            patch_attention(base_reader, os.path.join(policy, M.WEIGHTS), ad, meta)
            with open(flag(f"result.{k}.json") + ".tmp", "w") as f:
                json.dump({"loss": lsum, "trained_tokens": n_all, "seconds": round(time.time() - ts, 1),
                           "grad_norm": gn, "learning_rate": lr}, f)
            os.replace(flag(f"result.{k}.json") + ".tmp", flag(f"result.{k}.json"))
        if dist:
            dist.barrier()                                    # the policy file is rewritten before anyone reports parked
        touch(f"parked{a.rank}.{k}")

    if not R0:
        return 0
    fin = json.load(open(flag("finish.json")))
    step = int(fin.get("step", 0))
    if step <= 0:
        log("no training step was kept: nothing was learned, so no model folder was written")
        return 3
    del model
    gc.collect()
    ad = torch.load(flag(f"adapters/step{step}.pt"))
    export(a.model, a.out, ad, meta, fin.get("training_md", ""), log=log)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
    except Exception as e:                                # noqa: BLE001 -- the last line is what Dream shows
        open(flag("failed"), "w").close()
        traceback.print_exc(file=sys.stderr)
        msg = str(e).splitlines()[0][:300] if str(e) else type(e).__name__
        print(f"training failed on card {a.card}: {msg}", file=sys.stderr, flush=True)
        rc = 1
    sys.exit(rc)
