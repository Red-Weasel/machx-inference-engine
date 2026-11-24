#!/usr/bin/env python3
"""Fixtures for tests/unit/inkling2_ops_test.cpp: small synthetic inputs and the outputs of the numpy reference's own
functions (scripts/inkling2/ref_forward.py: bf16_round, urms, rms, l2clip, sconv_static, sconv_dynamic, attn_res, the
E2M1 / E4M3 tables, and the attention / router / expert lines of Inkling2.attention / Inkling2.moe written out on
plain arrays). No model file needed.

  ~/models/inkling2-reference/venv/bin/python -I scripts/inkling2/gen_ops_fixtures.py
writes tests/unit/inkling2_ops_fixtures.inc: {"name", {float, ...}} initialisers (shortest exact float32 spelling).
Integer arrays (ids, bytes) are stored as floats (exact for the small values used).
"""
import os, struct, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ref_forward as R  # noqa: E402

f32 = np.float32
rng = np.random.default_rng(20261122)
out = []


def put(name, a):
    a = np.ascontiguousarray(np.asarray(a, dtype=f32)).ravel()
    out.append((name, a))


def rnd(*shape, s=1.0):
    return (rng.standard_normal(shape) * s).astype(f32)


def bf16_vals(*shape, s=1.0):                    # values exactly representable in bf16 (weights stored as BF16)
    return R.bf16_round(rnd(*shape, s=s))


# 1. bf16 rounding, including exact ties (0x3F808000 -> even 0x3F80, 0x3F818000 -> 0x3F82)
x = np.concatenate([rnd(12, s=3.0), np.array([0x3F808000, 0x3F818000, 0xBF808000, 0x3F80FFFF], np.uint32).view(f32)])
put("bf16.x", x); put("bf16.y", R.bf16_round(x))

# 2. rmsnorm with / without weight
x = rnd(3, 40, s=2.0); w = bf16_vals(40)
put("rms.x", x); put("rms.w", w); put("rms.yw", R.rms(x, w)); put("rms.y", R.urms(x))

# 3. scale + L2 clip, rows = T * heads
x = rnd(2 * 3, 8, s=1.5); sc = bf16_vals(3, 8, s=0.8)
xs = (x.reshape(2, 3, 8) * sc[None]).reshape(6, 8)
put("clip.x", x); put("clip.scale", sc); put("clip.y", R.l2clip(xs, 2.0))

# 4. static short conv [T 7, C 5], weight [C, 1, 4]
x = rnd(7, 5); w = bf16_vals(5, 1, 4, s=0.5)
put("sconv.x", x); put("sconv.w", w.reshape(5, 4)); put("sconv.y", R.sconv_static(x, w))

# 5. dynamic short conv [T 7, C 8], 2 heads, dyn [T, 2 * 4], static_w [4, C]
x = rnd(7, 8); dyn = rnd(7, 8, s=0.3); sw = bf16_vals(4, 8, s=0.5)
put("dconv.x", x); put("dconv.dyn", dyn); put("dconv.sw", sw); put("dconv.y", R.sconv_dynamic(x, dyn, sw, 2))

# 6. attn_res: 2 block states + partial, [T 4, H 32], 3 queries
S = [rnd(4, 32, s=s) for s in (0.5, 3.0, 1.0)]; Q = rnd(3, 32, s=4.0)
o = R.attn_res(S[:2], S[2], Q)                                      # [T, nq, H]
for i, s in enumerate(S): put(f"ares.s{i}", s)
put("ares.q", Q); put("ares.y", o.transpose(1, 0, 2))                # [nq, T, H]

# 7. attention core (Inkling2.attention steps 5, 7, 8) on given q, k, v, relative states, proj, sink
T, nh, nkv, hd, dr, ext = 6, 4, 2, 8, 4, 4
q = rnd(T, nh, hd); k = rnd(T, nkv, hd); v = rnd(T, nkv, hd); rel = rnd(T, nh, dr, s=0.5); proj = rnd(dr, ext); sink = rnd(T, nh)
pos = np.arange(T)
dist = pos[:, None] - pos[None, :]
relb = rel @ proj                                                    # [T, nh, ext]
ok = (dist >= 0) & (dist < ext)                                      # extent 4 < T: the bias cut-off is exercised
bias = np.where(ok[:, None, :], np.take_along_axis(relb, np.broadcast_to(np.clip(dist, 0, ext - 1)[:, None, :], (T, nh, T)), axis=2), f32(0))
put("att.q", q); put("att.k", k); put("att.v", v); put("att.rel", rel); put("att.proj", proj); put("att.sink", sink)
for name, window in (("glob", 0), ("win3", 3)):
    g = nh // nkv
    kk, vv = np.repeat(k, g, axis=1), np.repeat(v, g, axis=1)
    s = np.einsum("thd,jhd->thj", q, kk) * f32(hd ** -0.5) + bias
    vis = dist >= 0
    if window:
        vis &= dist < window
    s = np.where(vis[:, None, :], s, f32(-np.inf))
    pr = R.softmax(np.concatenate([s, sink[:, :, None]], axis=-1))[..., :-1]
    put(f"att.o_{name}", np.einsum("thj,jhd->thd", pr, vv).reshape(T, nh * hd))

# 8. router (Inkling2.moe): E 16, k 8, an exact tie at the 8th / 9th place on token 1
E, K = 16, 8
lg = rnd(3, E, s=2.0); b = rnd(E, s=0.1).astype(f32)
sel = R.sigmoid(lg) + b
order = np.argsort(-sel[1], kind="stable")
lg[1, order[8]] = lg[1, order[7]]; b[order[8]] = b[order[7]]       # same score and bias: the lower id must win
scores = R.sigmoid(lg); sel = scores + b
own = np.sort(np.argsort(-sel, axis=1, kind="stable")[:, :K], axis=1)
forced = np.stack([rng.permutation(E)[:K] for _ in range(3)])       # unsorted on purpose
def weights(idx):
    sc = np.take_along_axis(scores, idx, axis=1)
    return sc / (sc.sum(axis=1, keepdims=True) + f32(1e-20)) * f32(8.0)
put("rt.logits", lg); put("rt.bias", b); put("rt.own", own); put("rt.w_own", weights(own))
put("rt.forced", forced); put("rt.forced_sorted", np.sort(forced, axis=1)); put("rt.w_forced", weights(np.sort(forced, axis=1)))
srt = -np.sort(-sel, axis=1)
put("rt.margin", srt[:, K - 1] - srt[:, K])

# 9. expert banks. NVFP4: E 3, rows 4, cols 32 (Weights.expert's decode); MXFP8: E 3, rows 4, cols 64
E, rows = 3, 4
valid = np.array([c for c in range(256) if (c & 0x7F) != 0x7F], np.uint8)
codes = rng.integers(0, 256, size=(E, rows, 16), dtype=np.uint8)
scl = rng.choice(valid[(valid & 0x78) != 0], size=(E, rows, 2))     # normal E4M3 scales
amax = np.abs(rnd(E, s=3.0)) + f32(0.5); den = f32(2688.0)
W = (R.NIB[codes].reshape(E, rows, 2, 16) * R.E4M3[scl][..., None]).reshape(E, rows, 32) * (amax / den)[:, None, None]
x = rnd(2, 32)                                                       # 2 tokens
idx = np.array([[2, 0], [1, 2]])                                     # k = 2 picks per token
put("nv.codes", codes); put("nv.scale", scl); put("nv.amax", amax); put("nv.x", x); put("nv.idx", idx)
put("nv.y", np.stack([[W[e] @ x[t] for e in idx[t]] for t in range(2)]))
vals = rng.choice(valid, size=(E, rows, 64)); ex = rng.integers(118, 136, size=(E, rows, 2), dtype=np.uint8)
W8 = (R.E4M3[vals].reshape(E, rows, 2, 32) * np.exp2(ex.astype(f32) - 127)[..., None]).reshape(E, rows, 64)
x8 = rnd(2 * 2, 64)                                                  # per (token, slot) inputs
put("mx.vals", vals); put("mx.exp", ex); put("mx.x", x8); put("mx.idx", idx)
put("mx.y", np.stack([[W8[idx[t][j]] @ x8[t * 2 + j] for j in range(2)] for t in range(2)]))

# 10. expert activation (SoftcappedSwiGLU), rows interleaved gate / up
gu = rnd(2, 12, s=6.0)
g, u = gu[:, 0::2], gu[:, 1::2]
put("act.gu", gu); put("act.y", f32(4) * np.tanh(g / f32(4)) * R.sigmoid(g) * f32(25) * np.tanh(u / f32(25)))

# 11. E4M3 table
put("e4m3", R.E4M3)

path = os.path.join(HERE, "..", "..", "tests", "unit", "inkling2_ops_fixtures.inc")
def lit(v):
    return "NAN" if np.isnan(v) else repr(float(v)) + "f" if float(v).is_integer() is False else f"{float(v):.1f}f"
with open(path, "w") as fh:
    fh.write("// GENERATED by scripts/inkling2/gen_ops_fixtures.py -- do not edit. {name, values} pairs for inkling2_ops_test.cpp.\n")
    for name, a in out:
        vals = ", ".join(lit(np.float32(v)) for v in a)
        fh.write(f'{{"{name}", {{{vals}}}}},\n')
print(f"wrote {len(out)} arrays to {os.path.normpath(path)}")
