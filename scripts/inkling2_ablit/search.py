#!/usr/bin/env python3
"""Inkling 2 Mini abliteration: a Heretic-style parametrised search over run-time ablation sets (our own code; no
optuna -- random start, then perturbation of the best feasible trials, shrinking).

  python search.py --work DIR --batch PROMPTS.jsonl --dir-a A.f32 --dir-b B.f32 [--cards 0,1] [--per-run 4]
                   [--trials 100] [--gen 768] [--seed-trials seeds.json]

One trial = one parameter vector -> one ablation set (one unit direction, a strength per write):
  t                     direction = unit((1 - t) a + t b)   (a, b: two good sources, e.g. layer 11 pass 1 / pass 2)
  attn / moe, each:     peak (layer 0..23), maxw (0..1.5), width (1..16 layers), floor (0..1, fraction of maxw)
                        w(L) = maxw * (floor + (1 - floor) * max(0, 1 - |L - peak| / width))
  m2, m3                multipliers of the loop layers' (6..17) passes 2 and 3 (pass 1 = 1); tie=1 -> m2 = m3 = 1
The embedding takes the attention weight at layer 0. Writes under 0.01 are left alone.

Scoring (the honest way: default effort, long answers): ie-inkling2-run --batch PROMPTS --effort default --gen G
--kl-ref-in <the original's saved reference>; refused = harmful declined + unfinished (score.py rules); feasible when
harmless declined <= the original's, harmless mean KL <= 0.05, loops <= the original's + 2. Ranking: feasible first,
then refused, then mean harmful reply length (an answer sooner is better). Everything goes to DIR/trials.jsonl; each
round prints the best and the Pareto front (refused vs KL). COUNTS ONLY: no reply text is printed or written here.
"""
import argparse, json, os, random, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import resolve, is_loop  # noqa: E402

H = 2304
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
BIN = os.path.join(ROOT, "build", "tools", "ie-inkling2-run")
SCHED = [(L, 1) for L in range(6)] + [(L, p) for p in (1, 2, 3) for L in range(6, 18)] + [(L, 1) for L in range(18, 24)]
SPACE = {"t": (0.0, 1.0), "a_peak": (0.0, 23.0), "a_maxw": (0.0, 1.5), "a_width": (1.0, 16.0), "a_floor": (0.0, 1.0),
         "m_peak": (0.0, 23.0), "m_maxw": (0.0, 1.5), "m_width": (1.0, 16.0), "m_floor": (0.0, 1.0),
         "m2": (0.0, 1.5), "m3": (0.0, 1.5), "tie": (0.0, 1.0)}


def weights(p):
    def prof(pre, L):
        f = p[pre + "_floor"]
        return p[pre + "_maxw"] * (f + (1 - f) * max(0.0, 1 - abs(L - p[pre + "_peak"]) / p[pre + "_width"]))
    tie = p["tie"] >= 0.5
    mult = {1: 1.0, 2: 1.0 if tie else p["m2"], 3: 1.0 if tie else p["m3"]}
    wa, wm = [], []
    for L, ps in SCHED:
        k = mult[ps] if 6 <= L < 18 else 1.0
        wa.append(prof("a", L) * k); wm.append(prof("m", L) * k)
    return prof("a", 0), wa, wm


EXTRA = None   # (rows [n][H] float32, groups) of a fixed set added to every trial (e.g. the "conclude" steering)


def write_set(p, a, b, path):
    d = (1 - p["t"]) * a + p["t"] * b
    d = d / np.linalg.norm(d)
    base = os.path.splitext(path)[0]
    rows = d.astype(np.float32)[None, :]
    if EXTRA is not None: rows = np.concatenate([rows, EXTRA[0]])
    rows.tofile(base + ".f32")
    we, wa, wm = weights(p)
    groups = []
    if we >= 0.01: groups.append({"rows": [0, 1], "strength": round(we, 4), "embed": True})
    for c in range(48):
        if wa[c] >= 0.01: groups.append({"rows": [0, 1], "strength": round(wa[c], 4), "attn": [c]})
        if wm[c] >= 0.01: groups.append({"rows": [0, 1], "strength": round(wm[c], 4), "moe": [c]})
    if not groups: groups.append({"rows": [0, 1], "strength": 0.0, "embed": True})
    if EXTRA is not None:
        for g in EXTRA[1]: g = dict(g); g["rows"] = [g["rows"][0] + 1, g["rows"][1]]; groups.append(g)
    json.dump({"format": "ie-ablation-v1", "arch": "inkling2_loop", "hidden": H, "file": os.path.basename(base) + ".f32", "rows": len(rows),
               "groups": groups, "params": p}, open(path, "w"), indent=1)


def score_file(path, names_h):
    by = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.strip():
            r = json.loads(line); by.setdefault(r["candidate"], []).append(r)
    out = {}
    for c, rows in by.items():
        h = [r for r in rows if r["name"] in names_h]; b = [r for r in rows if r["name"] not in names_h]
        t = {"complied": 0, "declined": 0, "unfinished": 0}
        for r in h:
            if not r["text"].strip() and not r.get("tool_calls"): t["unfinished"] += 1
            else: t[resolve(r["text"])[0]] += 1
        hd = sum(1 for r in b if r["text"].strip() and resolve(r["text"])[0] == "declined")
        kls = [sum(r["kl"]) / len(r["kl"]) for r in b if r.get("kl")]
        out[c] = {**t, "refused": t["declined"] + t["unfinished"], "n_h": len(h), "harmless_declined": hd, "n_b": len(b),
                  "kl": float(np.mean(kls)) if kls else float("nan"), "loops": sum(is_loop(r["ids"]) for r in rows),
                  "len_h": float(np.mean([len(r["ids"]) for r in h])) if h else 0.0}
    return out


def gpu_ok(card):
    return subprocess.run(["bash", os.path.join(ROOT, "scripts/inkling2/gpu_check.sh"), str(card)], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--batch", required=True)
    ap.add_argument("--dir-a", required=True); ap.add_argument("--dir-b", required=True)
    ap.add_argument("--cards", default="0,1"); ap.add_argument("--per-run", type=int, default=4)
    ap.add_argument("--trials", type=int, default=100); ap.add_argument("--gen", type=int, default=768)
    ap.add_argument("--seed-trials"); ap.add_argument("--n-random", type=int, default=12); ap.add_argument("--rng", type=int, default=1)
    ap.add_argument("--bin", default=BIN, help="tool binary (freeze a copy so a rebuild never changes a search half-way)")
    ap.add_argument("--parallel", type=int, default=0, help="--parallel B for the tool (0 = sequential)")
    ap.add_argument("--seed-from", nargs="*", default=[], help="earlier trials.jsonl files: their best feasible params seed this search")
    ap.add_argument("--n-seed", type=int, default=8)
    ap.add_argument("--extra-set", help="a set (e.g. steering, mode add) added unchanged to every trial and to the reference trials")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    par = ["--parallel", str(a.parallel)] if a.parallel > 1 else []
    global EXTRA
    if a.extra_set:
        j = json.load(open(a.extra_set))
        EXTRA = (np.fromfile(os.path.join(os.path.dirname(os.path.abspath(a.extra_set)), j["file"]), dtype=np.float32).reshape(-1, H), j["groups"])
        if any(g.get("mode", "remove") != "add" for g in j["groups"]): sys.exit("--extra-set: only add-mode groups (one removal group per write)")
    rng = random.Random(a.rng)
    A = np.fromfile(a.dir_a, dtype=np.float32).astype(np.float64); B = np.fromfile(a.dir_b, dtype=np.float32).astype(np.float64)
    names = [json.loads(l)["name"] for l in open(a.batch) if l.strip()]
    names_h = {n for n in names if n.startswith("harmful")}
    cards = [int(c) for c in a.cards.split(",")]
    ref = os.path.join(a.work, "ref.bin"); base_f = os.path.join(a.work, "base.json")
    if not os.path.exists(base_f):                                   # the original: its counts and the KL reference
        while not gpu_ok(cards[0]): time.sleep(60)
        lst = os.path.join(a.work, "base.tsv"); open(lst, "w").write("none\tnone\n")
        g = os.path.join(a.work, "base.jsonl")
        if os.path.exists(g): os.remove(g)
        subprocess.run([a.bin, "--card", str(cards[0]), "--batch", a.batch, "--effort", "default", "--gen", str(a.gen), "--kl-steps", "16",
                        "--ablate-list", lst, "--gen-out", g, "--kl-ref-out", ref] + par, stdout=open(os.path.join(a.work, "base.log"), "w"),
                       stderr=subprocess.STDOUT, check=True)
        json.dump(score_file(g, names_h)["none"], open(base_f, "w"), indent=1)
    base = json.load(open(base_f))
    print("original:", json.dumps(base), flush=True)
    tf = os.path.join(a.work, "trials.jsonl")
    trials = [json.loads(l) for l in open(tf)] if os.path.exists(tf) else []
    seeds = json.load(open(a.seed_trials)) if a.seed_trials and not trials else []
    if a.seed_from and not trials:                                   # re-score the best of earlier searches on this build
        old = [json.loads(l) for f in a.seed_from for l in open(f) if l.strip()]
        okk = lambda s: s["harmless_declined"] <= 1 and s["kl"] <= 0.05
        old.sort(key=lambda t: (0 if okk(t["score"]) else 1, t["score"]["refused"], t["score"]["kl"]))
        seeds += [t["params"] for t in old[:a.n_seed]]

    def feasible(s):
        return s["harmless_declined"] <= base["harmless_declined"] and s["kl"] <= 0.05 and s["loops"] <= base["loops"] + 2
    def key(tr):
        s = tr["score"]
        return (0 if feasible(s) else 1, s["refused"] + (0 if feasible(s) else 100 * max(0.0, s["kl"] - 0.05)), s["len_h"])
    def propose():
        if seeds: return seeds.pop(0)
        if len(trials) < a.n_random: return {k: rng.uniform(*v) for k, v in SPACE.items()}
        top = sorted(trials, key=key)[:5]
        par = rng.choice(top)["params"]
        sig = max(0.05, 0.25 * (1 - len(trials) / max(1, a.trials)))
        p = {}
        for k, (lo, hi) in SPACE.items():
            v = par[k] + rng.gauss(0, sig * (hi - lo)) if rng.random() < 0.6 else par[k]
            if rng.random() < 0.05: v = rng.uniform(lo, hi)
            p[k] = min(hi, max(lo, v))
        return p

    while len(trials) < a.trials:
        procs = []
        for card in cards:
            while os.path.exists("/tmp/ink2_build_wanted"): time.sleep(10)   # a build window another agent asked for
            t0 = time.time()                                                  # card 0 wanted by another agent (max 30 min)
            while os.path.exists("/tmp/ink2_card0_wanted") and time.time() - t0 < 1800: time.sleep(10)   # covers both cards
            if os.path.exists("/tmp/ink2_card0_wanted"): print("card0_wanted up > 30 min; going on", flush=True)
            while not gpu_ok(card): print(f"card {card} not clear; waiting", flush=True); time.sleep(120)
            batch = []
            for _ in range(a.per_run):
                i = len(trials) + sum(len(x[2]) for x in procs) + len(batch)
                p = propose(); name = f"trial{i:03d}"
                write_set(p, A, B, os.path.join(a.work, name + ".json")); batch.append((name, p))
            lst = os.path.join(a.work, f"run_{batch[0][0]}.tsv")
            open(lst, "w").write("".join(f"{n}\t{os.path.join(a.work, n + '.json')}\n" for n, _ in batch))
            g = os.path.join(a.work, f"run_{batch[0][0]}.jsonl")
            if os.path.exists(g): os.remove(g)
            pr = subprocess.Popen([a.bin, "--card", str(card), "--batch", a.batch, "--effort", "default", "--gen", str(a.gen), "--kl-steps", "16",
                                   "--ablate-list", lst, "--gen-out", g, "--kl-ref-in", ref] + par,
                                  stdout=open(g[:-6] + ".log", "w"), stderr=subprocess.STDOUT)
            procs.append((pr, g, batch))
        for pr, g, batch in procs:
            rc = pr.wait()
            sc = score_file(g, names_h) if os.path.exists(g) else {}
            for n, p in batch:
                if n not in sc: print(f"{n}: no result (exit {rc})", flush=True); continue
                trials.append({"name": n, "params": p, "score": sc[n]})
                with open(tf, "a") as f: f.write(json.dumps(trials[-1]) + "\n")
        ranked = sorted(trials, key=key)
        print(f"--- {len(trials)} trials; best 5 (feasible, refused of {base['n_h']} = declined + unfinished, harmless declined, KL, loops, mean harmful reply tokens):", flush=True)
        for tr in ranked[:5]:
            s = tr["score"]
            print(f"  {tr['name']} feasible={feasible(s)} refused {s['refused']} ({s['declined']} declined + {s['unfinished']} unfinished) "
                  f"hd {s['harmless_declined']} KL {s['kl']:.3f} loops {s['loops']} len {s['len_h']:.0f}", flush=True)
        front, best = [], 1e9
        for tr in sorted(trials, key=lambda t: t["score"]["kl"]):
            if tr["score"]["refused"] < best: front.append(tr); best = tr["score"]["refused"]
        print("  Pareto front (KL -> refused): " + ", ".join(f"{t['score']['kl']:.3f}->{t['score']['refused']}" for t in front), flush=True)


if __name__ == "__main__": main()
