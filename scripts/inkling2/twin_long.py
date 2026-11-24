#!/usr/bin/env python3
"""The numpy twin (ref_forward.py, fp32) on a long prompt, with given expert picks, for a tight engine cross-check.

ref_forward.Inkling2.moe decodes and caches every expert it touches (~97 GB once a long prompt touches them all) and loops
per token; this subclass groups the tokens of one call by expert and decodes each expert once per call without caching.
Same formulas (SPEC 7). fp32 only (no --bf16).

  python -I twin_long.py --ids-from META.json --routes ROUTES.json(step0 or [48][T][8]) --out DIR
writes DIR/twin_callNN.npy ([T, 2304] float32 per layer call) and DIR/twin_logits_last.npy.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_forward as R  # noqa: E402

f32 = np.float32


class Twin(R.Inkling2):
    def moe(self, L, x):
        c = self.cfg
        p = f"layers.{L}.mlp."
        scores = R.sigmoid(x @ self.P(p + "gate.weight").T)
        forced = np.sort(np.asarray(self.forced_routes[len(self.route_log)], dtype=int), axis=1)   # [T, 8]
        sc = np.take_along_axis(scores, forced, axis=1)
        wt = sc / (sc.sum(axis=1, keepdims=True) + f32(1e-20)) * f32(c["route_scale"])
        out = np.zeros_like(x)
        for e in np.unique(forced):
            toks, slots = np.nonzero(forced == e)
            gu = x[toks] @ self.w.expert(L, "w13", int(e)).T
            g, u = gu[:, 0::2], gu[:, 1::2]
            act = (f32(c["mlp_gate_softcap"]) * np.tanh(g / f32(c["mlp_gate_softcap"])) * R.sigmoid(g)
                   * f32(c["mlp_up_softcap"]) * np.tanh(u / f32(c["mlp_up_softcap"])))
            out[toks] += wt[toks, slots][:, None] * (act @ self.w.expert(L, "w2", int(e)).T)
        self.route_log.append((L, []))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-from", required=True)
    ap.add_argument("--routes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = json.load(open(a.ids_from))
    ids = meta.get("prompt_ids") or meta["ids"]
    r = json.load(open(a.routes))
    routes = r["step0"] if isinstance(r, dict) else r
    os.makedirs(a.out, exist_ok=True)
    m = Twin(R.MODEL)
    m.forced_routes = routes
    t0 = time.time()

    def on_call(i, L, out):
        np.save(os.path.join(a.out, f"twin_call{i:02d}.npy"), out.astype(f32))
        print(f"call {i:2d} layer {L:2d}  {time.time() - t0:.0f} s", flush=True)

    _, _, logits = m.forward(ids, on_call=on_call)
    np.save(os.path.join(a.out, "twin_logits_last.npy"), logits[-1])
    print(f"done in {time.time() - t0:.0f} s; top-1 {int(np.argmax(logits[-1]))}")


if __name__ == "__main__":
    main()
