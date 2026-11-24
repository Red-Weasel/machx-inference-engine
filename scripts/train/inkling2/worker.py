"""One training process on one card (machx_train.py starts one per card). Rank 0 also renders the data, writes the
metrics, the checkpoints and, at the end, the model folder.

Every step takes `batch` examples (default 8); with two cards each takes half (balanced by tokens), the adapter
gradients are summed across the cards (gloo, on the CPU: 27 MB a step) and both apply the same AdamW update, so the
two copies stay identical. Learning rate: linear warm-up, then cosine to 10 %.
A held-out set (--eval, else 3 % of the training sessions, 8 to 32, whole sessions) is scored before the first step,
every `eval_every` steps and at the end: loss (token-weighted) and eval_success_rate = the share of trained tokens of
the held-out assistant turns whose greedy next token is right. Training stops early when the held-out loss has not
improved (by 0.5 %) in `patience` evaluations; the folder gets the adapter of the best evaluation.
SIGTERM: the example under way finishes, that partial step is dropped, and the folder is written from what was learned. Nothing is ever printed
from the data.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import random
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
ap.add_argument("--data", required=True)
ap.add_argument("--eval", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--metrics", required=True)
ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--max-examples", type=int, default=0)
ap.add_argument("--max-len", type=int, default=12288)
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--world", type=int, default=1)
ap.add_argument("--card", type=int, required=True)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--lora-rank", type=int, default=16)
ap.add_argument("--alpha", type=float, default=32.0)
ap.add_argument("--eval-every", type=int, default=0, help="0: about 10 evaluations a run (at least every 5 steps)")
ap.add_argument("--eval-max", type=int, default=32)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--tools-sample", default="")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
R0 = a.rank == 0


def log(msg):
    print(f"[card {a.card}] {msg}", file=sys.stderr, flush=True)


def flag(name):
    return os.path.join(a.work, name)


def wait_for(path, what, timeout=3600):
    t = time.time()
    while not os.path.exists(path):
        if os.path.exists(flag("failed")):
            raise SystemExit(1)
        if STOP["flag"] and time.time() - t > 5:
            raise SystemExit(3)
        if time.time() - t > timeout:
            raise RuntimeError(f"timed out waiting for {what}")
        time.sleep(1)


def main():
    import torch
    from transformers import AutoTokenizer
    from inkling2 import data as D
    from inkling2 import model as M
    from inkling2.export import export

    t_start = time.time()
    dev = "xpu:0"                         # machx_train makes exactly one card visible to this process
    tok = AutoTokenizer.from_pretrained(a.model)

    # ------------------------------------------------------------------------------------- data (rank 0 renders)
    data_file = flag("data.pt")
    if R0 and not (a.resume and os.path.exists(data_file)):
        sp = a.tools_sample or D.newest_sample()
        if not sp:
            raise D.DataError("no Dream request sample found for the tool declarations (python -m dream.request_sample)")
        sample = D.load_sample(sp)
        tools = sample["tools"]
        train, st = D.load_file(a.data, tok, tools, a.max_len, max_examples=a.max_examples or None)
        held, est = [], None
        if a.eval:
            held, est = D.load_file(a.eval, tok, tools, a.max_len)
            recs = sorted({e["rec"] for e in held})
            random.Random(a.seed).shuffle(recs)
            keep = set(recs[:a.eval_max])
            held = [e for e in held if e["rec"] in keep]
        else:
            recs = sorted({e["rec"] for e in train})
            n = min(a.eval_max, max(8, round(0.03 * len(recs)))) if len(recs) >= 40 else 0
            random.Random(a.seed).shuffle(recs)
            ho = set(recs[:n])
            held = [e for e in train if e["rec"] in ho]
            train = [e for e in train if e["rec"] not in ho]
        if not train:
            raise D.DataError("the data file has no example to train on")
        tools_md5 = hashlib.md5(json.dumps(tools, sort_keys=True).encode()).hexdigest()[:8]
        page = sample.get("system", "")
        torch.save({"train": train, "held": held, "stats": st, "eval_stats": est,
                    "sample": {"file": os.path.abspath(sp), "taken_at": sample.get("taken_at"),
                               "tools": [t.get("function", t).get("name") for t in tools], "tools_md5": tools_md5,
                               "page_characters": len(page), "page_md5": hashlib.md5(page.encode()).hexdigest()[:8]}},
                   data_file + ".tmp")
        os.replace(data_file + ".tmp", data_file)
        log(f"data: {json.dumps(st)}")
    wait_for(data_file, "the rendered data")
    d = torch.load(data_file, weights_only=False)
    train, held = d["train"], d["held"]

    # --------------------------------------------------------------------- the model (loads staggered: host RAM peak)
    if a.rank:
        wait_for(flag(f"loaded{a.rank - 1}"), "the previous card's load")
    t0 = time.time()
    tok, model, tracker, abl = M.load(a.model, dev, log=log)
    open(flag(f"loaded{a.rank}"), "w").close()
    log(f"model on the card in {time.time() - t0:.0f} s ({torch.xpu.memory_allocated(dev) / 2**30:.1f} GiB)")
    adapters = M.add_lora(model, r=a.lora_rank, alpha=a.alpha, seed=a.seed)
    keys = sorted(adapters)
    params = [p for k in keys for p in (adapters[k].lora_A, adapters[k].lora_B)]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))

    dist = None
    if a.world > 1:
        import torch.distributed as dist
        dist.init_process_group("gloo", init_method="file://" + flag("dist_init"), rank=a.rank, world_size=a.world)

    def allsum(values):
        t = torch.tensor(values, dtype=torch.float64)
        if dist:
            dist.all_reduce(t)
        return t.tolist()

    per_pass = math.ceil(len(train) / a.batch)
    steps = min(a.max_steps, per_pass) if a.max_steps else per_pass
    warm = min(a.warmup, max(1, steps // 10))
    every = a.eval_every or max(5, math.ceil(steps / 10))

    def lr_at(step):
        if step <= warm:
            return a.lr * step / warm
        p = min(1.0, (step - warm) / max(1, steps - warm))
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    order = list(range(len(train)))
    random.Random(a.seed + 1).shuffle(order)

    def mine(items):
        """This card's share of a list of examples: longest first to the card with fewer tokens (same on every card)."""
        load, out = [0] * a.world, []
        for i in sorted(range(len(items)), key=lambda i: -len(items[i]["ids"])):
            r = min(range(a.world), key=lambda r: (load[r], r))
            load[r] += len(items[i]["ids"])
            if r == a.rank:
                out.append(items[i])
        return out

    def evaluate():
        if not held:
            return None
        tot = cor = n = 0.0
        with torch.no_grad():
            for e in mine(held):
                if STOP["flag"]:
                    break
                l, c, k = M.example_loss(model, tracker, e["ids"], e["labels"], dev)
                tot += l.item() * k
                cor += c
                n += k
        tot, cor, n, cut = allsum([tot, cor, n, 1.0 if STOP["flag"] else 0.0])
        if cut > 0 or n == 0:                      # a partial evaluation is not comparable
            return None
        return {"loss": tot / n, "success": cor / n}

    def adapter_tensors():
        return {**{k + ".A": adapters[k].lora_A.detach().cpu().clone() for k in keys},
                **{k + ".B": adapters[k].lora_B.detach().cpu().clone() for k in keys}}

    state = {"step": 0, "best": None, "best_step": 0, "since_best": 0, "evals": [], "losses": [],
             "train_seconds": 0.0, "train_tokens": 0}
    best_adapter = None
    sp = flag("state.pt")
    if a.resume and os.path.exists(sp):
        st = torch.load(sp, weights_only=False)
        with torch.no_grad():
            for k in keys:
                adapters[k].lora_A.copy_(st["A"][k])
                adapters[k].lora_B.copy_(st["B"][k])
        opt.load_state_dict(st["opt"])
        state, best_adapter = st["state"], st.get("best_adapter")
        log(f"resumed at step {state['step']}")

    def metrics(rec):
        if R0:
            with open(a.metrics, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def save_state():
        if R0:
            torch.save({"A": {k: adapters[k].lora_A.detach().cpu() for k in keys},
                        "B": {k: adapters[k].lora_B.detach().cpu() for k in keys}, "opt": opt.state_dict(),
                        "state": state, "best_adapter": best_adapter}, sp + ".tmp")
            os.replace(sp + ".tmp", sp)

    def do_eval(step):
        nonlocal best_adapter
        ev = evaluate()
        if ev is None:
            return False
        state["evals"].append({"step": step, **ev})
        if state["best"] is None or ev["loss"] < state["best"] * (1 - 0.005):
            state["best"], state["best_step"], state["since_best"] = ev["loss"], step, 0
            best_adapter = adapter_tensors()
        else:
            state["since_best"] += 1
        log(f"eval at step {step}: held-out loss {ev['loss']:.4f}, token accuracy {ev['success']:.4f}")
        return state["since_best"] >= a.patience

    if state["step"] == 0 and not state["evals"]:
        do_eval(0)
        e0 = state["evals"][-1] if state["evals"] else None
        metrics({"step": 0, "learning_rate": 0.0, "elapsed_seconds": round(time.time() - t_start, 1),
                 **({"eval_loss": e0["loss"], "eval_success_rate": e0["success"]} if e0 else {})})
    stopped_by = "the data (one pass)" if not a.max_steps or a.max_steps >= per_pass else f"--max-steps {a.max_steps}"
    last_save = time.time()
    for step in range(state["step"] + 1, steps + 1):
        if allsum([1.0 if STOP["flag"] else 0.0])[0] > 0:
            stopped_by = "a stop request"
            break
        ts = time.time()
        p0 = (step - 1) * a.batch
        batch = [train[order[(p0 + j) % len(train)]] for j in range(a.batch)]
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        lsum = toks = done = 0.0
        for e in mine(batch):
            if STOP["flag"]:                          # a step can outlast Dream's 300 s: stop between examples
                break
            try:
                loss, _, _ = M.example_loss(model, tracker, e["ids"], e["labels"], dev)
                (loss / a.batch).backward()
            except torch.OutOfMemoryError:            # an unusually heavy example: skipped and counted
                loss = None
                import gc
                gc.collect()
                torch.xpu.empty_cache()
                state["oom_skipped"] = state.get("oom_skipped", 0) + 1
                log(f"an example of {len(e['ids'])} tokens did not fit on the card and was skipped "
                    f"({state['oom_skipped']} so far)")
                if state["oom_skipped"] > max(3, 0.02 * len(train)):
                    raise RuntimeError("too many examples do not fit on the card; try a smaller --max-len")
                continue
            lsum += loss.item()
            toks += len(e["ids"])
            done += 1
        if allsum([1.0 if STOP["flag"] else 0.0])[0] > 0:   # the partial step is dropped on every card
            opt.zero_grad(set_to_none=True)
            stopped_by = "a stop request"
            break
        if dist:
            flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().cpu() for p in params])
            dist.all_reduce(flat)
            o = 0
            for p in params:
                p.grad = flat[o:o + p.numel()].view_as(p).to(p.device)
                o += p.numel()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        if not math.isfinite(gn):
            raise RuntimeError(f"the gradient became non-finite at step {step}")
        opt.step()
        lsum, toks, done = allsum([lsum, toks, done])
        lmean = lsum / max(1.0, done)
        dt = time.time() - ts
        state["step"], state["train_seconds"], state["train_tokens"] = step, state["train_seconds"] + dt, state["train_tokens"] + toks
        state["losses"].append(lmean)
        rate = state["train_tokens"] / state["train_seconds"]
        left = sum(len(train[order[(j) % len(train)]]["ids"]) for j in range(step * a.batch, steps * a.batch))
        rec = {"step": step, "loss": lmean, "learning_rate": lr_at(step), "grad_norm": gn,
               "elapsed_seconds": round(time.time() - t_start, 1), "tokens_per_second": round(toks / dt, 1),
               "eta_seconds": round(left / rate * 1.1)}          # +10 %: the evaluations
        early = False
        if step % every == 0 or step == steps:
            early = do_eval(step)
            if state["evals"] and state["evals"][-1]["step"] == step:
                rec["eval_loss"] = state["evals"][-1]["loss"]
                rec["eval_success_rate"] = state["evals"][-1]["success"]
            save_state()
        metrics(rec)
        if time.time() - last_save > 900:
            save_state()
            last_save = time.time()
        log(f"step {step}/{steps} loss {rec['loss']:.4f} lr {rec['learning_rate']:.2e} {dt:.0f} s {rec['tokens_per_second']:.0f} tok/s")
        if early:
            stopped_by = f"early stop: no held-out improvement in {a.patience} evaluations"
            break

    # --------------------------------------------------------------------------------------------- the folder
    if dist:
        dist.barrier()
    if not R0:
        return 0
    if state["step"] == 0:
        log("stopped before the first training step: nothing was learned, so no model folder was written")
        return 3
    if held and (not state["evals"] or state["evals"][-1]["step"] != state["step"]) and not STOP["flag"]:
        do_eval(state["step"])
    use_best = best_adapter is not None and state["best_step"] > 0
    tensors = best_adapter if use_best else adapter_tensors()
    used_step = state["best_step"] if use_best else state["step"]
    save_state()
    del model
    import gc
    gc.collect()
    torch.xpu.empty_cache()
    e0 = state["evals"][0] if state["evals"] else None
    eb = next((e for e in state["evals"] if e["step"] == used_step), None)
    L = state["losses"]
    first = sum(L[:5]) / len(L[:5])
    last = sum(L[-5:]) / len(L[-5:])
    s = d["sample"]
    md = f"""# Training record

- Trained: {datetime.date.today().isoformat()}, by machx-train (LoRA rank {a.lora_rank}, alpha {a.alpha:g}, attention q/k/v/o of all 24 layers, folded into the BF16 weights)
- Base model: {os.path.basename(os.path.abspath(a.model))}{" (its run-time ablation file was applied in the training forward and is copied here)" if abl else ""}
- Data file: {os.path.basename(a.data)}: {d['stats']['records']} sessions read, {len(train)} training examples (windows of at most {a.max_len} tokens), {len(held)} held-out examples ({'from ' + os.path.basename(a.eval) if a.eval else 'whole sessions held out of the data file'})
- Steps: {state['step']} of {steps} planned, {a.batch} examples a step, on {a.world} card(s); ended by {stopped_by}
- Training loss (mean of the first / last 5 steps): {first:.4f} -> {last:.4f}
- Held-out loss: {f"{e0['loss']:.4f} before -> {eb['loss']:.4f} at step {used_step}" if (e0 and eb) else (f"{e0['loss']:.4f} before; not scored again (stopped first)" if e0 else "no held-out set")}
- Held-out token accuracy (greedy next token on the trained turns): {f"{e0['success']:.4f} -> {eb['success']:.4f}" if (e0 and eb) else (f"{e0['success']:.4f} before; not scored again" if e0 else "-")}
- Weights saved from step {used_step} ({"the best held-out evaluation" if use_best else ("the last step: the held-out loss never improved on the base model" if len(state["evals"]) > 1 else "the last step")})
- Rendering: the model's chat template; tool declarations from the Dream request sample {os.path.basename(s['file'])} (taken {s['taken_at']}): tools {", ".join(s['tools'])} (md5 {s['tools_md5']}); each session's own system page (a Qwen tool preamble removed)
- Training time {state['train_seconds'] / 3600:.2f} h, {state['train_tokens'] / max(1, state['train_seconds']):.0f} tokens per second
"""
    export(a.model, a.out, tensors, {"scale": a.alpha / a.lora_rank, "keys": keys}, md, log=log)
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
        if "out of memory" in msg.lower() or type(e).__name__ == "OutOfMemoryError":
            msg = "the card ran out of memory; try a smaller --max-len"
        print(f"training failed on card {a.card}: {msg}", file=sys.stderr, flush=True)
        rc = 1
    sys.exit(rc)
