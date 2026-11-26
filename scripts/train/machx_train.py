#!/usr/bin/env python3
"""machx-train: fine-tune a local model on this machine's Intel Arc cards (the command Dream's Train view calls).

  machx-train --probe --model PATH
  machx-train --model PATH --data FILE.jsonl --out NEW_FOLDER --metrics FILE.jsonl [--eval FILE.jsonl]
              [--max-steps N] [--max-examples N] [--cards 0,1] [--max-len N] [--resume] [--tools-sample FILE]

A data file of sessions trains by supervised fine-tuning; a file of tasks whose lines carry a "reward" (grpo_rewards.py)
trains by GRPO, reinforcement learning on the model's own attempts scored by the reward (grpo.py).

Contract (Dream docs/training.md): the probe answers within seconds with one JSON line as the LAST stdout line
{"arch", "supported", "reason", "needs_gib_per_card"}; PATH is a checkpoint folder or a GGUF file. A run exits 0 with
NEW_FOLDER a complete model folder, or non-zero with a plain-words last stderr line. SIGTERM: the step under way
finishes and the folder is written from what was learned. The metrics file gets one JSON object per line.
See README.md beside this file.
"""
import argparse
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                  # python3 -I leaves the script's own folder off the path (grpo.py lives here)
ENGINE_ROOT = os.path.dirname(os.path.dirname(HERE))
VENV = os.path.expanduser("~/models/inkling2-train")            # torch 2.15 + XPU, transformers 5.24 with the Inkling 2 code
GPU_CHECK = os.path.join(ENGINE_ROOT, "scripts", "inkling2", "gpu_check.sh")

# What a supported architecture needs (measured, scripts/train/README.md).
SUPPORTED = {
    "inkling2_loop": {
        "name": "Inkling 2 Mini",
        "needs_gib_per_card": 30,          # 17.7 GiB of weights + activations at the default 12,288-token windows
        "host_gib_per_card": 72,           # the reference load peaks at 66 GiB of host memory, 24 GiB after it
    },
}


def fail(msg, code=1):
    print(msg, file=sys.stderr, flush=True)
    sys.exit(code)


# ------------------------------------------------------------------------------------------------------- probe
def gguf_arch(path):
    """general.architecture from a GGUF header (no tensor data read)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        ver = struct.unpack("<I", f.read(4))[0]
        if ver < 2:
            raise ValueError("GGUF version 1 is not read")
        f.read(8)                                           # tensor count
        n_kv = struct.unpack("<Q", f.read(8))[0]
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def rstr():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        def skip(t):
            if t == 8:
                rstr()
            elif t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                for _ in range(n):
                    skip(et)
            else:
                f.read(sizes[t])

        for _ in range(n_kv):
            key = rstr()
            t = struct.unpack("<I", f.read(4))[0]
            if key == "general.architecture" and t == 8:
                return rstr()
            skip(t)
    return None


def detect(path):
    """(arch, why-unsupported or None)."""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path):
        cfg = os.path.join(path, "config.json")
        if not os.path.exists(cfg):
            ggufs = [f for f in os.listdir(path) if f.endswith(".gguf")]
            if ggufs:
                return detect(os.path.join(path, sorted(ggufs)[0]))[0], "only checkpoint folders are trained, not GGUF files"
            return "unknown", "the folder has no config.json"
        c = json.load(open(cfg))
        arch = c.get("model_type") or (c.get("architectures") or ["unknown"])[0]
        if arch in SUPPORTED:
            need = ["model-00001-of-00001.safetensors", "tokenizer.json", "chat_template.jinja"]
            missing = [n for n in need if not os.path.exists(os.path.join(path, n))]
            if missing:
                return arch, f"the folder is incomplete (missing {', '.join(missing)})"
            return arch, None
        return arch, f"no trainer for {arch} yet"
    if os.path.isfile(path):
        try:
            arch = gguf_arch(path) or "unknown"
        except (OSError, ValueError, struct.error) as e:
            return "unknown", f"cannot read the file as a model ({e})"
        if arch in SUPPORTED:
            return arch, f"{arch} is trained from its checkpoint folder, not from a GGUF file"
        return arch, f"no trainer for {arch} yet"
    return "unknown", "no such file or folder"


def probe(path):
    arch, why = detect(path)
    ok = why is None
    rec = {"arch": arch, "supported": ok,
           "reason": (f"{SUPPORTED[arch]['name']} can be trained (LoRA on the attention weights) on one or two cards"
                      if ok else why),
           "needs_gib_per_card": SUPPORTED[arch]["needs_gib_per_card"] if arch in SUPPORTED else None}
    print(json.dumps(rec), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------- checks
def card_free(card):
    r = subprocess.run(["bash", GPU_CHECK, str(card)], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout.strip().splitlines() or [""])[-1].replace("gpu_check: ", "")


def available_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def quick_data_check(path, what):
    """Structure only (the worker renders and checks fully): JSON objects with a messages list. No text is shown."""
    if not os.path.isfile(path):
        fail(f"the {what} file does not exist")
    n = 0
    with open(path, encoding="utf-8", errors="strict") as f:
        try:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    fail(f"the {what} file is not JSON lines: line {n} does not parse")
                if not isinstance(r, dict) or not isinstance(r.get("messages"), list):
                    fail(f"the {what} file has a line ({n}) that is not an object with a 'messages' list")
        except UnicodeDecodeError:
            fail(f"the {what} file is not UTF-8 text")
    if n == 0:
        fail(f"the {what} file is empty")
    return n


# --------------------------------------------------------------------------------------------------------- run
def run(a):
    arch, why = detect(a.model)
    if why:
        fail(f"cannot train this model: {why}")
    spec = SUPPORTED[arch]
    model = os.path.abspath(os.path.expanduser(a.model))
    out = os.path.abspath(os.path.expanduser(a.out))
    if os.path.exists(out):
        fail("the output folder already exists")
    if not os.path.isdir(os.path.dirname(out)):
        fail("the output folder's parent folder does not exist")
    quick_data_check(a.data, "training data")
    if a.eval:
        quick_data_check(a.eval, "check")
    st = os.statvfs(os.path.dirname(out))
    free_gb = st.f_bavail * st.f_frsize / 1e9
    if free_gb < 20 + 21:                    # the copied weight file (19.5 GB) and 20 GB left for the system
        fail(f"not enough free disk space beside the output folder ({free_gb:.0f} GB free, 41 GB needed)")

    cards = [int(c) for c in a.cards.split(",")] if a.cards else [0, 1]
    busy = []
    for c in cards:
        ok, msg = card_free(c)
        if not ok:
            busy.append(f"card {c}: {msg}")
    if busy:
        fail("a graphics card is in use, so training did not start (" + "; ".join(busy) + ")")
    avail = available_gib()
    need = spec["host_gib_per_card"] + 26 * (len(cards) - 1) + 8      # loads are staggered: one peak + the others settled
    if avail < need:
        fail(f"not enough free memory on this computer to load the model ({avail:.0f} GiB available, {need} GiB needed)")

    import grpo
    if grpo.is_grpo_file(a.data):                        # tasks with rewards: reinforcement learning (grpo.py)
        if a.resume:
            fail("a GRPO run cannot be resumed yet; start it again")
        if free_gb < 20 + 21 + 20:
            fail(f"not enough free disk space beside the output folder for GRPO ({free_gb:.0f} GB free, 61 GB "
                 "needed: the result, the engine's working copy of the model, and 20 GB for the system)")
        if len(cards) != 2:
            fail("GRPO needs both cards (the engine writes the attempts on both)")
        return grpo.run(a, spec, model, out, cards, fail, VENV, HERE)

    work = out + ".work"
    if os.path.isdir(work) and not a.resume:
        shutil.rmtree(work)
    os.makedirs(work, exist_ok=True)
    for f in ("failed", "dist_init") + tuple(f"loaded{r}" for r in range(len(cards))):
        if os.path.exists(os.path.join(work, f)):
            os.remove(os.path.join(work, f))
    open(a.metrics, "a").close()

    stop = {"flag": False}

    def on_term(*_):
        stop["flag"] = True
        for p in procs:
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGTERM)
                except OSError:
                    pass
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    procs, logs = [], []
    for r, c in enumerate(cards):
        lp = os.path.join(work, f"worker{r}.log")
        logs.append(lp)
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={spec['host_gib_per_card']}G",
               "-p", "MemorySwapMax=0", "--",
               os.path.join(VENV, "xpu_env.sh"), os.path.join(VENV, "venv/bin/python"), "-I",
               os.path.join(HERE, "inkling2", "worker.py"),
               "--model", model, "--data", os.path.abspath(a.data), "--out", out, "--work", work,
               "--metrics", os.path.abspath(a.metrics), "--rank", str(r), "--world", str(len(cards)), "--card", str(c),
               "--max-len", str(a.max_len)]
        for k in ("eval", "max_steps", "max_examples", "tools_sample"):
            v = getattr(a, k)
            if v:
                cmd += ["--" + k.replace("_", "-"), os.path.abspath(v) if k in ("eval", "tools_sample") else str(v)]
        if a.resume:
            cmd.append("--resume")
        env = dict(os.environ, ZE_AFFINITY_MASK=str(c), PYTORCH_ALLOC_CONF="expandable_segments:True")
        procs.append(subprocess.Popen(cmd, stdout=open(lp, "a"), stderr=subprocess.STDOUT, env=env, cwd=VENV))
    print(f"training {spec['name']} on card(s) {a.cards or '0,1'}; logs in {work}", flush=True)
    dead_at = None
    while any(p.poll() is None for p in procs):           # a card's process that fails must not leave the other waiting
        if any(p.poll() not in (None, 0) for p in procs):
            dead_at = dead_at or time.time()
            if time.time() - dead_at > 60:
                for p in procs:
                    if p.poll() is None:
                        p.kill()
            elif time.time() - dead_at > 5:
                for p in procs:
                    if p.poll() is None:
                        p.send_signal(signal.SIGTERM)
        time.sleep(1)
    rcs = [p.returncode for p in procs]

    if rcs[0] == 0 and os.path.isdir(out):
        shutil.rmtree(work, ignore_errors=True)            # checkpoints and rendered token ids: no longer needed
        print(f"model written to {out}", flush=True)
        return 0
    # the plain reason: the last 'training failed' / stop line any worker wrote
    reason = None
    for lp in logs:
        for line in open(lp, errors="replace"):
            line = line.strip()
            if line.startswith("training failed") or "nothing was learned" in line:
                reason = re.sub(r"^\[card \d+\] ", "", line)
    if reason is None:
        if stop["flag"]:
            reason = "training was stopped before a model folder could be written"
        elif any(rc in (-9, 137) for rc in rcs):
            reason = "the training process was killed (out of host memory?)"
        else:
            reason = f"training failed (exit codes {rcs}); see {work}"
    fail(reason, 3 if stop["flag"] else 1)


def main():
    ap = argparse.ArgumentParser(prog="machx-train", description=__doc__.split("\n\n")[0])
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data")
    ap.add_argument("--eval", default="")
    ap.add_argument("--out")
    ap.add_argument("--metrics")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--cards", default="", help="cards to train on, e.g. 1 or 0,1 (default 0,1)")
    ap.add_argument("--max-len", type=int, default=12288, help="tokens per training window (default 12288)")
    ap.add_argument("--resume", action="store_true", help="continue the run whose work folder is <out>.work")
    ap.add_argument("--tools-sample", default="", help="a Dream request sample (python -m dream.request_sample) for "
                    "the tool declarations; default: the newest in the dream-sample folder")
    g = ap.add_argument_group("GRPO (used when the data file's lines carry a 'reward'; see grpo.py)")
    g.add_argument("--group", type=int, default=8, help="attempts at each task (default 8)")
    g.add_argument("--tasks-per-step", type=int, default=8, help="tasks a step (default 8)")
    g.add_argument("--max-new", type=int, default=6144, help="new tokens an attempt may write (default 6144)")
    g.add_argument("--lr", type=float, default=1e-5, help="learning rate (default 1e-5)")
    g.add_argument("--eval-attempts", type=int, default=3, help="attempts at each held-out task per check (default 3)")
    g.add_argument("--parallel", type=int, default=12, help="attempts the engine writes at once (default 12)")
    g.add_argument("--port", type=int, default=18695, help="the engine's port during the run (default 18695)")
    g.add_argument("--eval-every", type=int, default=0, help="held-out check every N steps (default: about 10 checks)")
    g.add_argument("--patience", type=int, default=3, help="stop after N held-out checks without improvement")
    g.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.probe:
        return probe(a.model)
    for k in ("data", "out", "metrics"):
        if not getattr(a, k):
            fail(f"--{k} is required for a training run")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
