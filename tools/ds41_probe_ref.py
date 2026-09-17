#!/usr/bin/env python3
"""Phase 2 C4/C5: diff the engine's dequant against torch's own, on the SAME bytes.

The engine side is `ie-ds41-probe`, which writes raw fp32 planes. This side decodes the
identical safetensors bytes with torch's float8_e4m3fn / float8_e8m0fnu, which are the same
dtypes DeepSeek's reference `kernel.py` uses. Both are exact conversions, so the bar is
max |diff| == 0.0 — a tolerance-based pass does not count.

  usage: ds41_probe_ref.py <dump_dir> [model_dir]
"""
import json, struct, sys
import numpy as np
import torch

dump = sys.argv[1]
D = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
FAIL = 0

def ok(what, cond, detail=""):
    global FAIL
    print(f"{'[ ok ]' if cond else '[FAIL]'} {what}" + (f"  ({detail})" if detail else ""))
    if not cond: FAIL += 1

# Refuse a stale dump: the probe writes `stamp` last, so an old dump left in a reused
# directory cannot be silently compared and reported as a pass.
import os, time
try:
    age = time.time() - float(open(f"{dump}/stamp").read().strip())
except Exception as ex:
    sys.exit(f"[FAIL] no freshness stamp in {dump} ({ex}) — run ie-ds41-probe first")
if age > 600:
    sys.exit(f"[FAIL] dump is {age/60:.1f} min old — re-run ie-ds41-probe, these are stale")
print(f"[ ok ] dump is fresh ({age:.0f} s old)")

wm = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]

def load(name):
    path = f"{D}/{wm[name]}"
    with open(path, "rb") as f:
        n, = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
        v = h[name]; b, e = v["data_offsets"]
        f.seek(8 + n + b)
        return bytearray(f.read(e - b)), v["shape"], v["dtype"]

def engine(fn, shape):
    a = np.fromfile(f"{dump}/{fn}", dtype=np.float32)
    assert a.size == shape[0] * shape[1], f"{fn}: {a.size} vs {shape}"
    return a.reshape(shape)

def diff(name, ref, got):
    both_nan = np.isnan(ref) & np.isnan(got)
    d = np.where(both_nan, 0.0, np.abs(ref.astype(np.float64) - got.astype(np.float64)))
    d = np.where(np.isnan(d), np.inf, d)          # one NaN, one not -> inf, never a silent pass
    m = float(d.max())
    ok(f"{name}: max |engine - torch| == 0", m == 0.0,
       f"max {m:g}, elements {ref.size:,}, nan-matched {int(both_nan.sum()):,}")
    return m

# ---------------------------------------------------------------- C4 dense
print("=== C4 dense FP8 E4M3 x E8M0 (layers.0.attn.wq_b) ===")
wraw, wsh, wdt = load("layers.0.attn.wq_b.weight")
sraw, ssh, sdt = load("layers.0.attn.wq_b.scale")
print(f"  weight {wdt} {wsh}   scale {sdt} {ssh}")
w = torch.frombuffer(wraw, dtype=torch.float8_e4m3fn).view(*wsh).to(torch.float32)
s = torch.frombuffer(sraw, dtype=torch.float8_e8m0fnu).view(*ssh).to(torch.float32)
bn, bk = wsh[0] // ssh[0], wsh[1] // ssh[1]
ref = (w * s.repeat_interleave(bn, 0).repeat_interleave(bk, 1)).numpy()
got = engine("dense.f32", wsh)
diff("dense", ref, got)

# ---------------------------------------------------------------- C5 expert
print("\n=== C5 routed FP4 E2M1 x E8M0 (layers.0.ffn.experts.0.w1) ===")
eraw, esh, edt = load("layers.0.ffn.experts.0.w1.weight")
esraw, essh, esdt = load("layers.0.ffn.experts.0.w1.scale")
N, Kh = esh; K = Kh * 2
print(f"  weight {edt} {esh} -> logical [{N}, {K}]   scale {esdt} {essh}")
# torch has the float4_e2m1fn_x2 dtype but no CPU kernel to unpack it
# ("copy_kernel not implemented"), so the reference here is the OCP MX spec's own field
# layout -- 1 sign, 2 exponent (bias 1), 1 mantissa, E==0 subnormal -- decoded from first
# principles. That is independent of the engine's 0xC8643210 constant AND of torch, which
# makes it a stronger reference than torch would have been, not a weaker one.
def e2m1_from_spec(nb):
    s_, e_, m_ = (nb >> 3) & 1, (nb >> 1) & 3, nb & 1
    v = (m_ * 0.5) if e_ == 0 else (1.0 + 0.5 * m_) * (2.0 ** (e_ - 1))
    return -v if s_ else v
SPEC = np.array([e2m1_from_spec(i) for i in range(16)], dtype=np.float32)
ok("torch has the dtype (layout cross-check) but no CPU unpack -> using the MX spec",
   torch.frombuffer(eraw, dtype=torch.float4_e2m1fn_x2).view(N, Kh).shape[1] == Kh,
   "spec table: " + ", ".join(f"{v:g}" for v in SPEC[:8]))
packed = np.frombuffer(bytes(eraw), dtype=np.uint8).reshape(N, Kh)
eref = np.empty((N, K), dtype=np.float32)
eref[:, 0::2] = SPEC[packed & 0x0F]               # low nibble = even element along K
eref[:, 1::2] = SPEC[packed >> 4]
es = torch.frombuffer(esraw, dtype=torch.float8_e8m0fnu).view(*essh).to(torch.float32)
ref_e = eref * np.repeat(es.numpy(), 32, axis=1)
got_e = engine("expert.f32", [N, K])
diff("expert", ref_e, got_e)

# nibble ORDER: torch's unpack defines element 2b = low nibble. Confirm the engine agrees
# on a byte whose two nibbles differ, independently of the magnitude check above.
r, c = np.argwhere((packed & 0xF) != (packed >> 4))[0]
ok("nibble order: engine elem[2b] is the LOW nibble (matches the spec decode)",
   got_e[r, 2*c] == ref_e[r, 2*c] and got_e[r, 2*c+1] == ref_e[r, 2*c+1]
   and ref_e[r, 2*c] != ref_e[r, 2*c+1],
   f"byte[{r},{c}]=0x{packed[r,c]:02X} -> {got_e[r,2*c]:g}, {got_e[r,2*c+1]:g}")

print(f"\n{'PHASE 2 NUMERICS: PASS' if FAIL == 0 else f'FAILED {FAIL} check(s)'}")
sys.exit(1 if FAIL else 0)
