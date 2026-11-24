#!/usr/bin/env python3
"""Compare an ie-inkling2-run --generate --dump-steps folder with a goldens2/<tag> folder (oracle_phase2.py).

  python -I compare_steps.py --dump DIR --golden GOLD [--tol-logits 8e-3]

Per step K (0 = the prompt forward, K >= 1 = after feeding the K-th generated id): last-position logit error, both
top-1s and the top-10 overlap. A step is only comparable while the engine was fed the same ids as the reference (always
true with --force-ids meta); after a free-running divergence the rest is reported but not judged. Then the generated ids
against the reference's, the first differing position with both logit margins there, and the expert picks of the engine
that differ from the reference's (sets per (step, call, token), with the engine's 8th-9th margin) before that point.
Exit status 0 when every comparable step's logits are within --tol-logits and every comparable top-1 matches.
"""
import argparse, json, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--tol-logits", type=float, default=8e-3)
    ap.add_argument("--tol-calls", type=float, default=5e-3, help="per-call bound when the dump has callsK.f32 (teacher-forced)")
    a = ap.parse_args()
    gm = json.load(open(os.path.join(a.golden, "meta.json")))
    groutes = json.load(open(os.path.join(a.golden, "routes.json")))
    dm = json.load(open(os.path.join(a.dump, "steps.json")))
    ref_gen, mine_gen = gm["generated"], dm["generated"]
    forced = dm.get("forced_ids") or []
    # ids fed before step K (K >= 1): the engine's own pick, or the forced id
    fed = [forced[k] if k < len(forced) else mine_gen[k] for k in range(len(mine_gen))]
    n = min(len(ref_gen), len(mine_gen))
    first_div = next((k for k in range(n) if ref_gen[k] != mine_gen[k]), None)
    first_fed_div = next((k for k in range(min(len(fed), len(ref_gen))) if fed[k] != ref_gen[k]), None)
    ok = True
    print(f"{'step':>4} {'relL2':>10} {'max|d|':>8} {'top1 mine':>10} {'top1 ref':>9} {'top10 overlap':>13}")
    for k in range(len(dm["steps"])):
        gf = os.path.join(a.golden, f"logits_step{k}.npy")
        if not os.path.exists(gf):
            break
        g = np.load(gf).astype(np.float32)
        m = np.fromfile(os.path.join(a.dump, f"step{k}.f32"), dtype=np.float32)[: g.size]
        comparable = first_fed_div is None or k <= first_fed_div
        rel = float(np.linalg.norm(m - g) / np.linalg.norm(g))
        tm, tg = np.argsort(-m, kind="stable")[:10], np.argsort(-g, kind="stable")[:10]
        flag = ""
        if comparable and (rel > a.tol_logits or tm[0] != tg[0]):
            # an exact tie in the bf16 golden logits is not a disagreement
            if not (tm[0] != tg[0] and g[tm[0]] == g[tg[0]] and rel <= a.tol_logits):
                ok = False; flag = "  <-- FAIL"
        if not comparable:
            flag = "  (inputs differ from the reference's: not judged)"
        print(f"{k:4d} {rel:10.3e} {np.abs(m - g).max():8.4f} {int(tm[0]):10d} {int(tg[0]):9d} {len(set(tm) & set(tg)):13d}{flag}")
    print("generated (engine):", mine_gen)
    print("generated (ref)   :", ref_gen)
    if first_div is None:
        print(f"generated ids: all {n} identical")
    else:
        k = first_div
        g = np.load(os.path.join(a.golden, f"logits_step{k}.npy")).astype(np.float32)
        m = np.fromfile(os.path.join(a.dump, f"step{k}.f32"), dtype=np.float32)[: g.size]
        r, e = ref_gen[k], mine_gen[k]
        print(f"first differing generated position: {k} (engine {e}, reference {r})")
        print(f"   reference logits there: ref token {g[r]:.4f}, engine token {g[e]:.4f}, margin {g[r] - g[e]:.4f}")
        print(f"   engine logits there:    ref token {m[r]:.4f}, engine token {m[e]:.4f}, margin {m[e] - m[r]:.4f}")
    # expert picks before the first divergence of the fed ids
    last = len(dm["steps"]) if first_fed_div is None else first_fed_div + 1
    diffs = []
    for k in range(min(last, len(dm["steps"]))):
        own = dm["steps"][k]["own"]
        ref = groutes.get(f"step{k}")
        if ref is None or not own or not own[0]:
            continue
        for c in range(48):
            for t in range(len(own[c])):
                if sorted(own[c][t]) != sorted(ref[c][t]):
                    diffs.append((k, c, t, dm["steps"][k]["margin"][c][t]))
    print(f"engine top-8 picks differing from the reference's (comparable steps): {len(diffs)}")
    for k, c, t, mg in diffs[:40]:
        print(f"   step {k:2d} call {c:2d} token {t:4d}: engine 8th-9th margin {mg:.2e}")
    # teacher-forced per-step calls (--teacher-steps): every layer call of every step against hidden_stepK.npy
    if os.path.exists(os.path.join(a.dump, "calls0.f32")):
        worst = (0.0, -1, -1)
        print("per-step worst layer call (teacher-forced):")
        for k in range(len(dm["steps"])):
            gf = os.path.join(a.golden, f"hidden_step{k}.npy")
            mf = os.path.join(a.dump, f"calls{k}.f32")
            if not (os.path.exists(gf) and os.path.exists(mf)):
                continue
            g = np.load(gf).astype(np.float32)
            m = np.fromfile(mf, dtype=np.float32).reshape(g.shape)
            rels = [float(np.linalg.norm(m[c] - g[c]) / np.linalg.norm(g[c])) for c in range(g.shape[0])]
            c = int(np.argmax(rels))
            print(f"   step {k:2d}: worst call {c:2d} relL2 {rels[c]:.3e}")
            if rels[c] > worst[0]:
                worst = (rels[c], k, c)
        print(f"teacher-forced worst: step {worst[1]} call {worst[2]} relL2 {worst[0]:.3e} (bound {a.tol_calls:g})")
        ok = ok and worst[0] <= a.tol_calls
    print("RESULT:", "PASS" if ok else "FAIL", f"(comparable steps: logits <= {a.tol_logits:g} and same top-1"
          + (f"; teacher-forced calls <= {a.tol_calls:g})" if os.path.exists(os.path.join(a.dump, "calls0.f32")) else ")"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
