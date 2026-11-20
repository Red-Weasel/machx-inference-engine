#!/usr/bin/env python3
"""Hy4-flash: the routed experts as an 8-bit pack the engine maps at load (no conversion, no minutes of RAM building).

  python pack_q8.py <source_dir> <ie_dir> [--layers 1-41] [--chunk 32]

Writes <ie_dir>/experts-q8-LNN.bin, one file per MoE layer: 512 slots in Hy4fQ8Store's layout
  [gate qs | up qs | down qs | gate d | up d | down d]     qs int8 [N][K], d fp16 bits [N][K / 32]
with the store's own rounding rule: d = fp16(max|w| / 127) of each 32 weights of a row (stepped up one fp16 when the
rounded scale would not reach the largest weight), qs = round(w / d) clamped to +-127. 7,520,256 bytes an expert, 3.59 GiB
a layer, 147 GiB in all. A layer whose file already has the right size is skipped; the writer waits when the target
drive is short of space. Run with the reference venv.
"""
import argparse, glob, json, os, shutil, struct, sys, time
import numpy as np
import torch
from safetensors import safe_open

def shard_index(src):
    out = {}
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
        for k in h:
            if k != "__metadata__": out[k] = f
    return out

def q8(w):
    """w [n, rows, K] fp32 -> (qs int8 [n, rows, K], d fp16-bit int16 [n, rows, K / 32])"""
    n, rows, K = w.shape
    b = w.reshape(n, rows, K // 32, 32)
    amax = b.abs().amax(-1)
    d16 = (amax / 127.0).to(torch.float16)
    short = d16.float() * 127.0 < amax
    bits = d16.view(torch.int16) + short.to(torch.int16)
    dv = bits.view(torch.float16).float()
    inv = torch.where(dv > 0, 1.0 / dv, torch.zeros_like(dv))
    qs = torch.round(b * inv.unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
    return qs.reshape(n, rows, K), bits

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--layers", default=""); ap.add_argument("--chunk", type=int, default=32)
    a = ap.parse_args()
    src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
    cfg = json.load(open(os.path.join(src, "config.json")))["text_config"]
    nL, E, MI, LAT = cfg["num_hidden_layers"], cfg["n_routed_experts"], cfg["moe_intermediate_size"], cfg["routed_expert_hidden_size"]
    moe = [L for L in range(nL) if cfg["mlp_layer_types"][L] == "sparse"]
    if a.layers:
        want = set()
        for part in a.layers.split(","):
            if "-" in part: x, y = part.split("-"); want.update(range(int(x), int(y) + 1))
            else: want.add(int(part))
        moe = [L for L in moe if L in want]
    gq, dq, gd, dd = MI * LAT, LAT * MI, MI * (LAT // 32) * 2, LAT * (MI // 32) * 2
    slot = (2 * gq + dq + 2 * gd + dd + 63) // 64 * 64
    idx = shard_index(src); t_all = time.time()
    for L in moe:
        out = os.path.join(dst, f"experts-q8-L{L:02d}.bin")
        if os.path.exists(out) and os.path.getsize(out) == E * slot: print(f"layer {L}: exists, skipped", flush=True); continue
        while shutil.disk_usage(dst).free < E * slot + (4 << 30):
            print(f"layer {L}: waiting for {E * slot / 2**30:.1f} GiB of space on the target drive", flush=True); time.sleep(30)
        gu, dn = f"model.layers.{L}.mlp.experts.gate_up_proj", f"model.layers.{L}.mlp.experts.down_proj"
        t0 = time.time()
        with safe_open(idx[gu], framework="pt") as fg, safe_open(idx[dn], framework="pt") as fd, open(out + ".tmp", "wb") as fo:
            sg, sd = fg.get_slice(gu), fd.get_slice(dn)
            assert sg.get_shape() == [E, 2 * MI, LAT] and sd.get_shape() == [E, LAT, MI]
            for e0 in range(0, E, a.chunk):
                e1 = min(E, e0 + a.chunk); n = e1 - e0
                qg, bg = q8(sg[e0:e1].float()); qd, bd = q8(sd[e0:e1].float())
                buf = np.zeros((n, slot), dtype=np.uint8)
                buf[:, 0:gq] = qg[:, :MI].reshape(n, gq).numpy().view(np.uint8)
                buf[:, gq:2 * gq] = qg[:, MI:].reshape(n, gq).numpy().view(np.uint8)
                buf[:, 2 * gq:2 * gq + dq] = qd.reshape(n, dq).numpy().view(np.uint8)
                o = 2 * gq + dq
                buf[:, o:o + gd] = bg[:, :MI].reshape(n, gd // 2).numpy().view(np.uint8).reshape(n, gd)
                buf[:, o + gd:o + 2 * gd] = bg[:, MI:].reshape(n, gd // 2).numpy().view(np.uint8).reshape(n, gd)
                buf[:, o + 2 * gd:o + 2 * gd + dd] = bd.reshape(n, dd // 2).numpy().view(np.uint8).reshape(n, dd)
                fo.write(buf.tobytes())
        os.replace(out + ".tmp", out)
        print(f"layer {L}: {E} experts -> {os.path.getsize(out) / 2**30:.2f} GiB, {time.time() - t0:.0f} s", flush=True)
    done = [L for L in range(nL) if cfg["mlp_layer_types"][L] == "sparse" and os.path.exists(os.path.join(dst, f"experts-q8-L{L:02d}.bin"))]
    json.dump({"format": "hy4flash-q8soa-1", "slot_bytes": slot, "experts": E, "moe_inter": MI, "latent": LAT, "layers": done, "source_dir": src},
              open(os.path.join(dst, "q8pack.json"), "w"), indent=1)
    print(f"8-bit pack: {len(done)} layers present, {(time.time() - t_all) / 60:.1f} min", flush=True)

if __name__ == "__main__": main()
