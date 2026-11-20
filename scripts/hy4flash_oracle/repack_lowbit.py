#!/usr/bin/env python3
"""Hy4-flash: an 8-bit-layout expert file whose integers fit b bits -> the engine's native b-bit file (Hy4fQFmt).

  python repack_lowbit.py <ie_dir_with_q8_layout_file> <out_dir> --layers 20 --bits 5,5 [--experts 512] [--inter 1536] [--latent 1536]

pack_lowbit_trial.py rounds experts to b bits (plainly or with GPTQ) but stores them in the 8-bit slot layout so the
8-bit path can measure them. This rewrites those very integers at their real width:
  8 bits  int8 [K]
  5 bits  nibbles [K / 2] then one high-bit word per 32 weights [K / 8]      (q + 16)
  4 bits  nibbles [K / 2]                                                    (q + 8)
gate / up at the first width, down at the second; the fp16 scales are copied. Output <out_dir>/experts-q<g><d>-LNN.bin
(experts-q8-LNN.bin for 8,8). The integers are checked against the width before anything is written.
"""
import argparse, os, sys, time
import numpy as np

def row_bytes(bits, K): return K if bits == 8 else (K // 2 + K // 8 if bits == 5 else K // 2)

def pack_rows(q, bits):
    """q int8 [n, R, K] -> uint8 [n, R, row_bytes]"""
    if bits == 8: return q.view(np.uint8)
    qmax = (1 << (bits - 1)) - 1
    if int(np.abs(q).max()) > qmax: raise SystemExit(f"an integer of magnitude {int(np.abs(q).max())} does not fit {bits} bits")
    v = (q.astype(np.int16) + (16 if bits == 5 else 8)).astype(np.uint8)
    nib = (v[..., 0::2] & 15) | ((v[..., 1::2] & 15) << 4)
    if bits == 4: return nib
    hb = np.packbits((v >> 4) & 1, axis=-1, bitorder="little")
    return np.concatenate([nib, hb], axis=-1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src_dir"); ap.add_argument("out_dir"); ap.add_argument("--layers", required=True); ap.add_argument("--bits", required=True)
    ap.add_argument("--experts", type=int, default=512); ap.add_argument("--inter", type=int, default=1536); ap.add_argument("--latent", type=int, default=1536); ap.add_argument("--chunk", type=int, default=16)
    a = ap.parse_args()
    bg, bd = (int(x) for x in a.bits.split(","))
    E, MI, LAT = a.experts, a.inter, a.latent
    gq8, dq8, gd, dd = MI * LAT, LAT * MI, MI * (LAT // 32) * 2, LAT * (MI // 32) * 2
    slot8 = (2 * gq8 + dq8 + 2 * gd + dd + 63) // 64 * 64
    gq, dq = MI * row_bytes(bg, LAT), LAT * row_bytes(bd, MI)
    slot = (2 * gq + dq + 2 * gd + dd + 63) // 64 * 64
    os.makedirs(a.out_dir, exist_ok=True)
    for L in (int(x) for x in a.layers.split(",")):
        t0 = time.time()
        src = os.path.join(a.src_dir, f"experts-q8-L{L:02d}.bin")
        if os.path.getsize(src) != E * slot8: raise SystemExit(f"{src}: not an 8-bit-layout file of {E} experts")
        name = f"experts-q8-L{L:02d}.bin" if (bg, bd) == (8, 8) else f"experts-q{bg}{bd}-L{L:02d}.bin"
        out = os.path.join(a.out_dir, name)
        m = np.memmap(src, dtype=np.uint8, mode="r").reshape(E, slot8)
        with open(out + ".tmp", "wb") as fo:
            for e0 in range(0, E, a.chunk):
                blk = np.asarray(m[e0:e0 + a.chunk]); n = len(blk)
                g = blk[:, 0:gq8].view(np.int8).reshape(n, MI, LAT); u = blk[:, gq8:2 * gq8].view(np.int8).reshape(n, MI, LAT)
                d = blk[:, 2 * gq8:2 * gq8 + dq8].view(np.int8).reshape(n, LAT, MI)
                buf = np.zeros((n, slot), dtype=np.uint8)
                buf[:, 0:gq] = pack_rows(g, bg).reshape(n, gq); buf[:, gq:2 * gq] = pack_rows(u, bg).reshape(n, gq)
                buf[:, 2 * gq:2 * gq + dq] = pack_rows(d, bd).reshape(n, dq)
                buf[:, 2 * gq + dq:2 * gq + dq + 2 * gd + dd] = blk[:, 2 * gq8 + dq8:2 * gq8 + dq8 + 2 * gd + dd]
                fo.write(buf.tobytes())
        os.replace(out + ".tmp", out)
        print(f"layer {L}: {src} -> {out} ({os.path.getsize(out) / 2**30:.2f} GiB, {slot} bytes an expert), {time.time() - t0:.0f} s", flush=True)

if __name__ == "__main__": main()
