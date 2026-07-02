#!/usr/bin/env python3
# tools/exl3/make_oracle.py — CUDA-free EXL3 decode oracle (ground truth generator).
#
# Reproduces exllamav3's EXL3 weight decode on CPU in pure numpy (no torch, no CUDA),
# faithfully from the source (file:line in docs/exl3_format_notes.md §2-§4,§7):
#   - bitstream → 16-bit code: exl3_dq.cuh:15-31 (tail-biting per 16x16 tile)
#   - codebook cb=0:           codebook.cuh:25-35   (x*mul+add ; (x&m)^c ; f16(lo)+f16(hi))
#   - tensor-core tile perm:   quantize.py:21-48    (lane-order → row-major)
#   - Hadamard reconstruction: quantize.py:244-262  (128-pt Sylvester, 1/sqrt(128), both sides)
#
# Emits raw little-endian binaries (trivial for the C++ host-decode test to read) +
# meta.json. Primary Task-1 oracle = onelayer.wrot.f16 (decode only). Full materialized
# weight (onelayer.weight.f16) is the Task-2 Hadamard+scale cross-check.
#
# Usage: tools/exl3/make_oracle.py <model.safetensors> <linear_base_key> <out_dir>

import sys, json, glob, os
import numpy as np
from safetensors import safe_open

CB0_MUL = np.uint32(89226354)
CB0_ADD = np.uint32(64248484)
CB0_AND = np.uint32(0x8fff8fff)
CB0_XOR = np.uint32(0x3b603b60)


def tensor_core_perm():
    """quantize.py:21-48 — lane-order index t (0..255) → row-major index (r*16+c)."""
    perm = np.zeros(256, dtype=np.int64)
    for t in range(32):
        r0 = (t % 4) * 2; r1 = r0 + 1; r2 = r0 + 8; r3 = r0 + 9
        c0 = t // 4;      c1 = c0 + 8
        perm[t*8+0] = r0*16 + c0
        perm[t*8+1] = r1*16 + c0
        perm[t*8+2] = r2*16 + c0
        perm[t*8+3] = r3*16 + c0
        perm[t*8+4] = r0*16 + c1
        perm[t*8+5] = r1*16 + c1
        perm[t*8+6] = r2*16 + c1
        perm[t*8+7] = r3*16 + c1
    return perm


def decode_cb0(code_u32):
    """codebook.cuh:25-35 cb=0, vectorized. lop3 0x6a = (a & b) ^ c (hand-verified)."""
    x = (code_u32.astype(np.uint32) * CB0_MUL + CB0_ADD).astype(np.uint32)
    x = ((x & CB0_AND) ^ CB0_XOR).astype(np.uint32)
    lo = (x & np.uint32(0xffff)).astype(np.uint16).view(np.float16).astype(np.float32)
    hi = (x >> np.uint32(16)).astype(np.uint16).view(np.float16).astype(np.float32)
    return (lo + hi)  # the two fp16 halves summed (__hadd)


def sylvester_norm(n):
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(n)).astype(np.float32)


def decode_trellis(trellis):
    """trellis int16 (TK, TN, 16*bits) → W_rot fp16 [K, N] (decode + un-permute, no Hadamard)."""
    TK, TN, w16 = trellis.shape
    bits = w16 // 16
    psz = bits * 256 // 32                              # u32 words per tile
    # int16 tile buffers → u32 words (little-endian native on x86): (TK,TN,psz)
    words = trellis.reshape(TK, TN, w16).view(np.uint16).reshape(TK, TN, w16) \
                   .view(np.uint32).reshape(TK, TN, psz)
    codes = np.zeros((256, TK, TN), dtype=np.uint32)
    for t in range(256):
        b0 = t * bits + bits - 16 + 256 * bits
        b1 = b0 + 16
        i0 = b0 // 32
        i1 = (b1 - 1) // 32
        s0 = (i1 + 1) * 32 - b1
        a = words[:, :, i0 % psz].astype(np.uint64)
        b = words[:, :, i1 % psz].astype(np.uint64)
        merged = (a << np.uint64(32)) | b               # fshift: (a<<32)|b
        codes[t] = ((merged >> np.uint64(s0)) & np.uint64(0xffff)).astype(np.uint32)
    dec = decode_cb0(codes)                             # (256, TK, TN) fp32
    perm = tensor_core_perm()
    tile_rm = np.zeros((TK, TN, 256), dtype=np.float32)
    tile_rm[:, :, perm] = np.transpose(dec, (1, 2, 0))  # rowmajor[perm[t]] = decoded[t]
    K, N = TK * 16, TN * 16
    W = tile_rm.reshape(TK, TN, 16, 16).transpose(0, 2, 1, 3).reshape(K, N)
    return W.astype(np.float16), bits


def materialize(W_rot, suh, svh):
    """Full fp16 weight: had_l(W_rot)*suh then had_r*svh (quantize.py get_weight_tensor)."""
    K, N = W_rot.shape
    H = sylvester_norm(128)
    w = W_rot.astype(np.float32)
    w = (H @ w.reshape(K // 128, 128, N)).reshape(K, N)         # preapply_had_l
    w = w * suh.astype(np.float32)[:, None]
    w = (w.reshape(K, N // 128, 128) @ H).reshape(K, N)         # preapply_had_r
    w = w * svh.astype(np.float32)[None, :]
    return w.astype(np.float16)


def main():
    model_path, base, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    f = safe_open(model_path, framework="numpy")
    keys = set(f.keys())
    trellis = f.get_tensor(base + ".trellis")
    suh = f.get_tensor(base + ".suh")
    svh = f.get_tensor(base + ".svh")
    has_mcg = (base + ".mcg") in keys
    has_mul1 = (base + ".mul1") in keys
    assert not (has_mcg or has_mul1), "cb!=0 not handled by this oracle (cb=0 only)"
    W_rot, bits = decode_trellis(trellis)
    K, N = W_rot.shape
    W_full = materialize(W_rot, suh, svh)

    trellis.tofile(os.path.join(out_dir, "onelayer.trellis"))      # int16 LE
    suh.astype(np.float16).tofile(os.path.join(out_dir, "onelayer.suh.f16"))
    svh.astype(np.float16).tofile(os.path.join(out_dir, "onelayer.svh.f16"))
    W_rot.tofile(os.path.join(out_dir, "onelayer.wrot.f16"))       # decode-only oracle
    W_full.tofile(os.path.join(out_dir, "onelayer.weight.f16"))    # full materialized
    meta = {
        "source_key": base, "bits": int(bits), "cb": 0,
        "K": int(K), "N": int(N),
        "tile_k": int(trellis.shape[0]), "tile_n": int(trellis.shape[1]),
        "words16_per_tile": int(trellis.shape[2]),
        "u32_per_tile": int(bits * 256 // 32),
        "had_dim": 128, "codebook_scale_folded_in_suh": True,
        "files": {
            "trellis": "onelayer.trellis (int16 LE [tile_k,tile_n,16*bits])",
            "suh": "onelayer.suh.f16 (fp16 [K])", "svh": "onelayer.svh.f16 (fp16 [N])",
            "wrot": "onelayer.wrot.f16 (fp16 [K,N] row-major; decode only)",
            "weight": "onelayer.weight.f16 (fp16 [K,N]; full had+scale)",
        },
    }
    with open(os.path.join(out_dir, "onelayer.meta.json"), "w") as mf:
        json.dump(meta, mf, indent=2)

    # Sanity: decoded values are a pseudo-Gaussian codebook → small, finite, non-constant.
    print(f"[oracle] {base}  bits={bits}  K={K} N={N}")
    print(f"[oracle] W_rot   : min={W_rot.min():.4f} max={W_rot.max():.4f} "
          f"mean={W_rot.astype(np.float32).mean():.5f} std={W_rot.astype(np.float32).std():.4f}")
    print(f"[oracle] W_full  : min={W_full.min():.4f} max={W_full.max():.4f} "
          f"std={W_full.astype(np.float32).std():.5f}  finite={np.isfinite(W_full.astype(np.float32)).all()}")
    print(f"[oracle] wrote → {out_dir}/onelayer.*")


if __name__ == "__main__":
    main()
