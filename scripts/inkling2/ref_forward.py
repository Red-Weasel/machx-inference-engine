#!/usr/bin/env python3
"""Inkling 2 Mini (inkling2_loop) text model: numpy-only reference forward pass, written from docs/inkling2/SPEC.md.

Full sequence at once, no KV cache. Compares the hidden state after each of the 48 layer calls, and the final logits,
with the publisher's reference values ("goldens").

  python -I ref_forward.py                       # fp32 arithmetic, exact decoded expert weights
  python -I ref_forward.py --bf16                # also round to bf16 wherever the reference does (section 9 of the spec)
  python -I ref_forward.py --teacher-force       # feed the golden output of call i-1 into call i: per-call error, not accumulated
  python -I ref_forward.py --routes R.json       # free-running, but with the reference's top-8 expert picks ([48][T][8])
  python -I ref_forward.py --ids 1,2,3 --no-compare

Exit status: 1 if any call's relative L2 error exceeds --tol (default 2e-2) or the top-1 token differs; else 0.

Precision. Default = fp32 arithmetic on the checkpoint's values. The reference runs in bf16: every linear / matmul result,
every norm / clip / conv / gate output and each layer's output are rounded to bf16 (R(...) below marks those points; --bf16
turns them on), its expert weights are the decoded values rounded to bf16, and its NVFP4 per-expert factor uses bf16(amax)
(--ref-amax). What that costs is measured in SPEC.md section 16. No torch here: numpy only.

Section numbers in the comments are those of SPEC.md.
"""
import argparse, glob, json, os, struct, sys
import numpy as np

MODEL = os.path.expanduser("~/models/Inkling-2-Mini")
GOLD_DIRS = [os.path.expanduser("~/models/inkling2-reference/goldens")]
f32 = np.float32


# ----------------------------------------------------------------------------------------------- weights
def bf16_round(x):
    """Round fp32 to the nearest bf16 (ties to even), returned as fp32. Finite inputs only."""
    u = np.ascontiguousarray(x, dtype=f32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return u.view(f32)


def e4m3_table():
    """FP8 E4M3 (the `e4m3fn` variant): 1 sign, 4 exponent (bias 7), 3 mantissa; exponent 0 = subnormal; no infinities;
    0x7F / 0xFF = NaN. Section 10.3."""
    t = np.zeros(256, dtype=f32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        v = (m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
        if e == 15 and m == 7:
            v = float("nan")
        t[b] = -v if s else v
    return t


E4M3 = e4m3_table()
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=f32)   # 10.1
NIB = np.stack([E2M1[np.arange(256) & 15], E2M1[np.arange(256) >> 4]], axis=-1)                   # [256, 2]: low nibble first


class Weights:
    """Memory-mapped safetensors. get() returns fp32 for BF16 / F32 tensors; raw() the stored bytes."""

    def __init__(self, path, prefix="model.llm."):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.h = json.loads(f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.prefix = prefix

    def has(self, name):
        return self.prefix + name in self.h

    def raw(self, name):
        e = self.h[self.prefix + name]
        a, b = e["data_offsets"]
        return self.mm[self.base + a:self.base + b], e["dtype"], e["shape"]

    def get(self, name):
        buf, dt, shape = self.raw(name)
        if dt == "BF16":
            u = np.frombuffer(buf, dtype="<u2").astype(np.uint32) << np.uint32(16)
            return u.view(f32).reshape(shape)
        if dt == "F32":
            return np.frombuffer(buf, dtype="<f4").reshape(shape).copy()
        raise ValueError(f"{name}: {dt}")

    def rows(self, name, idx):
        """Selected rows of a BF16 [N, C] matrix, as fp32."""
        buf, dt, shape = self.raw(name)
        assert dt == "BF16"
        u = np.frombuffer(buf, dtype="<u2").reshape(shape)[idx].astype(np.uint32) << np.uint32(16)
        return u.view(f32)

    def expert(self, layer, which, e, ref_amax=False):
        """Decoded weight matrix of one expert, fp32, checkpoint row order. Section 10.
        which = 'w13' -> [2304 rows (gate/up interleaved), 2304]; 'w2' -> [2304, 1152].
        ref_amax: use bf16(amax) / bf16(denominator) as the per-expert NVFP4 factor. The reference's loader casts
        every source tensor to bf16 before its converter runs, so that is the factor the goldens were made with
        (up to 0.39 % off the checkpoint's fp32 amax, a different amount per expert). Section 16."""
        p = f"layers.{layer}.mlp.experts.{which}_weight"
        wbuf, wdt, wshape = self.raw(p)
        sbuf, sdt, sshape = self.raw(p + "_scale")
        rows = wshape[1]
        w = np.frombuffer(wbuf, dtype=np.uint8).reshape(wshape)[e]
        s = np.frombuffer(sbuf, dtype=np.uint8).reshape(sshape)[e]
        if self.has(p + "_amax"):                                   # NVFP4 (10.1)
            assert wdt == "U8" and sdt == "F8_E4M3"
            amax = self.get(p + "_amax")[e]
            den = self.get(p + "_outer_scale_denominator")[0]
            vals = NIB[w].reshape(rows, sshape[2], 16)              # element 2i = low nibble of byte i
            vals = vals * E4M3[s][..., None]                        # one scale per 16 columns of a row
            if ref_amax:
                amax, den = bf16_round(np.array([amax], f32))[0], bf16_round(np.array([den], f32))[0]
            vals = vals * (amax / den)                              # one factor per expert
        else:                                                       # MXFP8 (10.2)
            assert wdt == "F8_E4M3" and sdt == "U8"
            vals = E4M3[w].reshape(rows, sshape[2], 32)
            vals = vals * np.exp2(s.astype(f32) - 127)[..., None]   # one exponent per 32 columns of a row
        return np.ascontiguousarray(vals.reshape(rows, -1), dtype=f32)


# ----------------------------------------------------------------------------------------------- primitives (section 1)
def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(f32)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def urms(x, eps=1e-6):                                              # 1.2
    return x * (1.0 / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + f32(eps))).astype(f32)


def rms(x, w, eps=1e-6):                                            # 1.1
    return w * urms(x, eps)


def l2clip(x, thr):                                                 # 1.3
    n = np.sqrt(np.sum(np.square(x), axis=-1, keepdims=True))
    return x * np.minimum(f32(thr) / (n + f32(1e-6)), f32(1.0))


def sconv_static(x, w):                                             # 1.8   x [T, C], w [C, 1, 4]
    T, K = x.shape[0], w.shape[-1]
    xp = np.concatenate([np.zeros((K - 1, x.shape[1]), f32), x], axis=0)
    conv = np.zeros_like(x)
    for j in range(K):                                              # w[:, 0, K-1] multiplies the current token
        conv += w[:, 0, j][None, :] * xp[j:j + T]
    return (conv + x) / f32(np.sqrt(K + 1))


def sconv_dynamic(x, dyn, static_w, heads=8):                       # 1.9   x [T, C], dyn [T, heads * 4], static_w [4, C]
    T, C = x.shape
    K = static_w.shape[0]
    W = static_w.T.reshape(heads, C // heads, K)[None] + dyn.reshape(T, heads, 1, K)       # [T, heads, C/heads, K]
    xp = np.concatenate([np.zeros((K - 1, C), f32), x], axis=0)
    conv = np.zeros((T, heads, C // heads), f32)
    for lag in range(K):                                            # lag 0 = current token
        conv += W[..., lag] * xp[K - 1 - lag:K - 1 - lag + T].reshape(T, heads, C // heads)
    return (conv.reshape(T, C) + x) / f32(np.sqrt(2 * K + 1))


def attn_res(states, partial, Q):                                   # 1.10  states: list of [T, H]; Q [nq, H] -> [T, nq, H]
    S = urms(np.stack(list(states) + [partial], axis=1).astype(f32))                       # [T, n, H]
    a = np.einsum("qh,tnh->tqn", Q, S) * f32(S.shape[-1] ** -0.5)
    return np.einsum("tqn,tnh->tqh", softmax(a), S)


# ----------------------------------------------------------------------------------------------- the model
class Inkling2:
    def __init__(self, model_dir=MODEL, bf16=False, expert_bf16=None, ref_amax=None, verbose=False):
        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))["text_config"]
        self.w = Weights(os.path.join(model_dir, "model-00001-of-00001.safetensors"))
        c = self.cfg
        self.H, self.NL = c["hidden_size"], c["num_hidden_layers"]
        self.block = c["n_layers_per_attn_res_block"]
        self.ls, self.le, self.passes = c["loop_config"]["core_start"], c["loop_config"]["core_end"], c["loop_config"]["num_passes"]
        self.bf16 = bf16
        self.expert_bf16 = bf16 if expert_bf16 is None else expert_bf16
        self.ref_amax = bf16 if ref_amax is None else ref_amax
        self.R = bf16_round if bf16 else (lambda x: x)              # the reference's "cast back to bf16" points
        self.verbose = verbose
        self._cache, self._ecache = {}, {}
        self.route_log = []
        self.taps = None                                            # set to {} to record intermediates (debugging)
        self.forced_routes = None                                   # [48][T][8] expert ids: use these picks (--routes)
        self._tapkey = None

    def tap(self, name, x):
        if self.taps is not None:
            self.taps[self._tapkey + (name,)] = np.array(x)

    def P(self, name):
        if name not in self._cache:
            self._cache[name] = self.w.get(name)
        return self._cache[name]

    def is_global(self, L):                                         # section 0
        return (L + 1) % self.block == 0

    def is_loop(self, L):
        return self.ls <= L < self.le

    def expert(self, L, which, e):
        k = (L, which, int(e))
        if k not in self._ecache:
            m = self.w.expert(L, which, int(e), self.ref_amax)
            self._ecache[k] = bf16_round(m) if self.expert_bf16 else m
        return self._ecache[k]

    def lin(self, x, name):
        return self.R(x @ self.P(name).T)

    # ---- section 3
    def attention(self, L, q_in, k_in, v_in, pos):
        c, R, glob = self.cfg, self.R, self.is_global(L)
        p = f"layers.{L}.attn."
        T = q_in.shape[0]
        nh, nkv, hd, dr = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["d_rel"]
        tau = (1.0 + c["log_scaling_alpha"] * np.log(np.maximum((pos + 1).astype(f32) / c["log_scaling_n_floor"], 1.0))).astype(f32)

        q = self.lin(q_in, p + "wq_du.weight").reshape(T, nh, hd)
        q = R(l2clip(q * self.P(p + "q_l2_clip.scale"), c["qk_l2_clip_threshold"]))
        if glob:
            q = R(q * tau[:, None, None])

        def kv(x_in, wname, cname, is_key):
            x = self.lin(x_in, p + wname)
            if glob:
                dyn = self.lin(x_in, p + cname + ".weight_proj")
                x = R(sconv_dynamic(x, dyn, self.P(p + cname + ".static_w"), nkv))
            else:
                x = R(sconv_static(x, self.P(p + cname + ".weight")))
            x = x.reshape(T, nkv, hd)
            return R(l2clip(x, c["qk_l2_clip_threshold"])) if is_key else R(urms(x, c["rms_norm_eps"]))

        k = kv(k_in, "wk_dv.weight", "k_sconv", True)
        v = kv(v_in, "wv_dv.weight", "v_sconv", False)
        self.tap("q", q); self.tap("k", k); self.tap("v", v)

        r = self.lin(q_in, p + "wr_du.weight").reshape(T, nh, dr)
        r = R(l2clip(r * self.P(p + "rel_l2_clip.scale"), c["rel_l2_clip_threshold"]))
        if glob:
            r = R(r * tau[:, None, None])
        proj = self.P(p + "rel_logits_proj.proj")                   # [d_rel, extent]
        extent = proj.shape[1]
        rel = R(r @ proj)                                           # [T, nh, extent]
        dist = pos[:, None] - pos[None, :]                          # [Tq, Tk]
        ok = (dist >= 0) & (dist < extent)
        bias = np.where(ok[:, None, :], np.take_along_axis(rel, np.broadcast_to(np.clip(dist, 0, extent - 1)[:, None, :], (T, nh, T)), axis=2), f32(0))

        sink = self.lin(q_in, p + "attn_sink_proj.weight")          # [T, nh]
        self.tap("bias", bias); self.tap("sink", sink)

        g = nh // nkv
        kk, vv = np.repeat(k, g, axis=1), np.repeat(v, g, axis=1)   # query head h reads KV head h // g
        s = R(np.einsum("thd,jhd->thj", q, kk))
        s = R(s * f32(hd ** -0.5))
        s = R(s + bias)
        vis = dist >= 0
        if not glob:
            vis &= dist < c["sliding_window_size"]
        s = np.where(vis[:, None, :], s, f32(-np.inf))
        pr = softmax(np.concatenate([s, sink[:, :, None]], axis=-1))
        pr = R(pr[..., :-1])                                        # the sink column is dropped
        o = R(np.einsum("thj,jhd->thd", pr, vv)).reshape(T, nh * hd)
        o = R(o * sigmoid(self.lin(q_in, p + "gate_attn.weight")))
        return self.lin(o, p + "wo_ud.weight")

    # ---- section 7
    def moe(self, L, x):
        c, R = self.cfg, self.R
        p = f"layers.{L}.mlp."
        logits = x @ self.P(p + "gate.weight").T                    # fp32, no rounding
        self.tap("router_logits", logits)
        scores = sigmoid(logits)
        sel_key = scores + self.P(p + "gate.bias")
        out = np.zeros_like(x)
        log = []
        forced = None if self.forced_routes is None else self.forced_routes[len(self.route_log)]
        for t in range(x.shape[0]):
            own = np.sort(np.argsort(-sel_key[t], kind="stable")[:c["num_experts_per_tok"]])
            idx = own if forced is None else np.sort(np.asarray(forced[t], dtype=int))
            wt = scores[t, idx] / (scores[t, idx].sum() + f32(1e-20)) * f32(c["route_scale"])
            margin = float(np.sort(sel_key[t])[-c["num_experts_per_tok"]] - np.sort(sel_key[t])[-c["num_experts_per_tok"] - 1])
            log.append((idx.tolist(), margin, own.tolist()))
            for e, we in zip(idx, wt):
                gu = R(self.expert(L, "w13", e) @ x[t])             # checkpoint rows: even = gate, odd = up
                gate, up = gu[0::2], gu[1::2]
                act = R(f32(c["mlp_gate_softcap"]) * np.tanh(gate / f32(c["mlp_gate_softcap"])) * sigmoid(gate)
                        * f32(c["mlp_up_softcap"]) * np.tanh(up / f32(c["mlp_up_softcap"])))
                out[t] += we * R(self.expert(L, "w2", e) @ act)
        self.route_log.append((L, log))
        self.tap("experts", R(out))
        return R(out)

    # ---- 1.11
    def gate(self, L, which, h, m):
        p = f"layers.{L}.controllers.{which}."
        g = self.R(np.concatenate([urms(h), urms(m)], axis=-1))
        g = self.lin(g, p + "down_weight")
        forget = np.tanh(g @ self.P(p + "forget_weight").T)
        gain = np.tanh(g @ self.P(p + "gain_weight").T)
        return (1 - forget) * h + gain * m

    # ---- 5.3
    def messages(self, C_prev, D_prev, i):
        a, b = self.P("middle_cycle_cross_pass_queries"), self.P("middle_cycle_cross_pass_kv_queries")
        Q = np.stack([a[2 * i], b[2 * i], b[2 * i + 1], a[2 * i + 1]])                 # q, k, v, mlp
        m = attn_res([C_prev], D_prev, Q)
        return m[:, 0], m[:, 1], m[:, 2], m[:, 3]

    # ---- section 4 (all four layer classes)
    def layer(self, L, h, BS, pos, msg=None):
        R, glob = self.R, self.is_global(L)
        p = f"layers.{L}."
        block_residual = h
        if L > 0:
            a = attn_res(BS, h, self.P(f"attn_res.attn_queries.{L - 1}").reshape(-1, self.H))
        else:
            a = h[:, None, :]
        if glob:
            aq, ak, av = a[:, 0], a[:, 1], a[:, 2]
            nq, nk, nv = p + "q_input_norm.weight", p + "k_input_norm.weight", p + "v_input_norm.weight"
        else:
            aq = ak = av = a[:, 0]
            nq = nk = nv = p + "attn_norm.weight"
        if msg is not None:
            aq, ak, av = self.gate(L, "attn", aq, msg[0]), self.gate(L, "k", ak, msg[1]), self.gate(L, "v", av, msg[2])
        q_in, k_in, v_in = R(rms(aq, self.P(nq))), R(rms(ak, self.P(nk))), R(rms(av, self.P(nv)))
        self.tap("q_in", q_in); self.tap("k_in", k_in); self.tap("v_in", v_in)

        x = self.attention(L, q_in, k_in, v_in, pos)
        self.tap("attn", x)
        x = R(sconv_static(x, self.P(p + "attn_sconv.weight")))
        self.tap("attn_sconv", x)
        if L % self.block == 0:                                     # block boundary: the input moves to the block list
            BS = BS + [block_residual]
            residual = x
        else:
            residual = block_residual + x
        u = attn_res(BS, residual, self.P(f"attn_res.mlp_queries.{L}").reshape(-1, self.H))[:, 0]
        if msg is not None:
            u = self.gate(L, "mlp", u, msg[3])
        self.tap("mlp_readout", u)
        mi = R(rms(u, self.P(p + "mlp_norm.weight")))
        self.tap("mlp_in", mi)
        y = self.moe(L, mi)
        y = R(sconv_static(y, self.P(p + "mlp_sconv.weight")))
        self.tap("mlp_sconv", y)
        return R(residual + y)

    # ---- sections 2, 5, 8
    def forward(self, ids, force=None, on_call=None):
        """force: optional list of 48 golden arrays; when given, call i's output is replaced by force[i] after it is
        recorded (teacher forcing). Returns (list of 48 call outputs, final hidden [T, H], logits [T, vocab])."""
        R = self.R
        T = len(ids)
        pos = np.arange(T)
        x0 = R(rms(self.w.rows("embed.weight", np.asarray(ids)), self.P("embed_norm.weight")))
        calls = []

        def run(L, h, BS, msg=None):
            self._tapkey = (len(calls),)
            out = self.layer(L, h, BS, pos, msg)
            calls.append(out)
            if on_call:
                on_call(len(calls) - 1, L, out)
            return force[len(calls) - 1].astype(f32) if force is not None else out

        h, BS = x0, []
        for L in range(0, self.ls):
            new = run(L, h, BS)
            if L % self.block == 0:
                BS = BS + [h]
            h = new
        pre_h, pre_BS = h, BS
        npre = len(pre_BS)
        for L in range(self.ls, self.le):                           # pass 1
            new = run(L, h, BS)
            if L % self.block == 0:
                BS = BS + [h]
            h = new
        for _ in range(1, self.passes):                             # passes 2..: restart from the pre-loop state
            prev_BS, prev_h = BS[npre + 1:], h
            assert len(prev_BS) == 1
            h, BS = pre_h, pre_BS
            for i, L in enumerate(range(self.ls, self.le)):
                new = run(L, h, BS, self.messages(prev_BS[0], prev_h, i))
                if L % self.block == 0:
                    BS = BS + [h]
                h = new
        for L in range(self.le, self.NL):
            new = run(L, h, BS)
            if L % self.block == 0:
                BS = BS + [h]
            h = new
        f = attn_res(BS, h, self.P("attn_res.output_query").reshape(-1, self.H))[:, 0]
        f = R(rms(f, self.P("norm.weight")))
        V = self.cfg["unpadded_vocab_size"]
        buf, _, shape = self.w.raw("unembed.weight")
        U = np.frombuffer(buf, dtype="<u2").reshape(shape)
        logits = np.empty((T, V), f32)
        for a in range(0, V, 16384):
            b = min(V, a + 16384)
            logits[:, a:b] = f @ (U[a:b].astype(np.uint32) << np.uint32(16)).view(f32).T
        logits = R(logits)
        cap = f32(self.cfg["final_logit_softcapping"])
        return calls, f, cap * np.tanh(logits / cap)


# ----------------------------------------------------------------------------------------------- comparison
def golden_file(d, i):
    """The hidden state returned by layer call i: call{NN}_layer{LL}_o.npy (a layer returns one bare tensor), or the
    older name call{NN}_layer{LL}.npy."""
    for pat in (f"call{i:02d}_layer[0-9][0-9]_o.npy", f"call{i:02d}_layer[0-9][0-9].npy"):
        hit = sorted(glob.glob(os.path.join(d, pat)))
        if hit:
            return hit[0]
    return None


def find_goldens(arg):
    for d in ([arg] if arg else GOLD_DIRS):
        if os.path.exists(os.path.join(d, "meta.json")) and all(golden_file(d, i) for i in range(48)):
            return d
    sys.exit(f"no complete goldens (48 call files + meta.json) in {arg or GOLD_DIRS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--goldens", default=None)
    ap.add_argument("--ids", default=None, help="comma-separated token ids (default: the goldens' prompt)")
    ap.add_argument("--bf16", action="store_true", help="round to bf16 at every point where the reference does")
    ap.add_argument("--expert-bf16", choices=["0", "1"], default=None, help="round the decoded expert weights to bf16 (default: same as --bf16)")
    ap.add_argument("--ref-amax", choices=["0", "1"], default=None,
                    help="NVFP4 per-expert factor from bf16(amax), as the reference's loader does (default: same as --bf16)")
    ap.add_argument("--teacher-force", action="store_true", help="feed each call the golden output of the previous call")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--print-routes", action="store_true", help="print the experts picked per call and the top-8 selection margin")
    ap.add_argument("--routes", default=None, help="JSON [48][T][8]: take the experts picked per call and token from the reference "
                    "(weights are still computed here); isolates the discrete top-8 choice from everything else")
    a = ap.parse_args()

    gold = meta = None
    if not a.no_compare:
        gd = find_goldens(a.goldens)
        meta = json.load(open(os.path.join(gd, "meta.json")))
        gold = [np.load(golden_file(gd, i)).astype(f32).reshape(-1, 2304) for i in range(48)]
        gl = [int(os.path.basename(golden_file(gd, i))[12:14]) for i in range(48)]
        print(f"goldens: {gd}")
    ids = [int(t) for t in a.ids.split(",")] if a.ids else meta["ids"]
    m = Inkling2(a.model, bf16=a.bf16, expert_bf16=None if a.expert_bf16 is None else a.expert_bf16 == "1",
                 ref_amax=None if a.ref_amax is None else a.ref_amax == "1")
    if a.routes:
        m.forced_routes = json.load(open(a.routes))
    print(f"ids {ids}   mode: {'bf16-rounding emulated' if a.bf16 else 'fp32'}, expert weights {'rounded to bf16' if m.expert_bf16 else 'exact'}, NVFP4 amax {'bf16 (as the reference)' if m.ref_amax else 'fp32 (checkpoint)'}"
          f"{', teacher-forced' if a.teacher_force else ''}{', expert picks forced from ' + a.routes if a.routes else ''}")
    worst = [0.0, -1]
    bad = False

    def on_call(i, L, out):
        nonlocal bad
        if gold is None:
            print(f"call {i:2d} layer {L:2d}  l2 {np.linalg.norm(out):.4f}", flush=True)
            return
        assert gl[i] == L, (i, L, gl[i])
        g = gold[i]
        d = out - g
        rel = float(np.linalg.norm(d) / np.linalg.norm(g))
        if rel > worst[0]:
            worst[:] = [rel, i]
        flag = "" if rel <= a.tol else "   <-- FAIL"
        bad |= rel > a.tol
        print(f"call {i:2d} layer {L:2d}  max|d| {np.abs(d).max():.5f}  relL2 {rel:.3e}  (golden l2 {np.linalg.norm(g):.3f}){flag}", flush=True)

    calls, f, logits = m.forward(ids, force=gold if a.teacher_force else None, on_call=on_call)
    if a.print_routes:
        for (L, log), i in zip(m.route_log, range(48)):
            print(f"call {i:2d} layer {L:2d} min top-8 margin {min(x[1] for x in log):.2e}  last token experts {log[-1][0]}")
    if a.routes:
        diff = [(i, L, t, x[1]) for i, (L, log) in enumerate(m.route_log) for t, x in enumerate(log) if x[0] != x[2]]
        print(f"forced routes: this implementation's own top-8 differs from the reference's in {len(diff)} of {48 * len(ids)} (call, token) pairs"
              + (f"; own 8th-vs-9th score margins there: max {max(d[3] for d in diff):.2e}, median {float(np.median([d[3] for d in diff])):.2e}" if diff else ""))
        for i, L, t, mg in diff:
            print(f"   call {i:2d} layer {L:2d} token {t}: own margin {mg:.2e}")
    top = np.argsort(-logits[-1], kind="stable")[:10]
    print("top-10 (last position):", [(int(t), round(float(logits[-1][t]), 4)) for t in top])
    if gold is None:
        return 0
    print(f"worst call: {worst[1]} relL2 {worst[0]:.3e}")
    gt = [t[0] for t in meta["top10"]]
    glast = np.load(os.path.join(gd, "logits_last.npy"))
    d = logits[-1] - glast
    print(f"final logits (last position): max|d| {np.abs(d).max():.5f}  relL2 {np.linalg.norm(d) / np.linalg.norm(glast):.3e}")
    print("golden top-10:", [(t[0], round(t[1], 4)) for t in meta["top10"]])
    same_set, same_order, top1 = set(gt) == set(top.tolist()), gt == top.tolist(), gt[0] == int(top[0])
    # the golden logits are bf16 values (the reference's unembed runs in bf16), so neighbours can be exactly tied there
    tie_order = all(glast[int(t)] == glast[g] for t, g in zip(top, gt))
    order = "same" if same_order else ("same up to exact ties in the golden" if tie_order else "different")
    print(f"top-1 {'same' if top1 else 'DIFFERENT'}; top-10 set {'same' if same_set else 'DIFFERENT'}; top-10 order {order}")
    if os.path.exists(os.path.join(gd, "logits_all.npy")):
        ga = np.load(os.path.join(gd, "logits_all.npy"))
        print("argmax per position: mine", logits.argmax(-1).tolist(), "golden", ga.argmax(-1).tolist(),
              f"  all-position logits relL2 {np.linalg.norm(logits - ga) / np.linalg.norm(ga):.3e}")
    ok = (not bad) and top1
    print("RESULT:", "PASS" if ok else "FAIL", f"(tolerance {a.tol:g} per call, top-1 must match)")
    if bad and not a.routes and not a.teacher_force:
        print("note: a free-running failure with every teacher-forced call passing means a near-tied top-8 expert pick went the\n"
              "      other way somewhere (SPEC.md section 16); re-run with --routes <reference picks> or --teacher-force to see.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
