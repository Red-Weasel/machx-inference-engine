#!/usr/bin/env python3
"""Hy4-flash: the released full-precision checkpoint -> the engine's model directory.

  python pack.py <source_dir> <ie_dir> [--layers 3,7 | 1-41] [--chunk 16] [--no-experts]

<source_dir> holds Tencent's BF16 safetensors shards (or a mini model's model.safetensors) and config.json. It is read,
never written. <ie_dir> receives:

  hy4flash_ie.json           what the engine cannot derive: the source directory, the n-gram hash multipliers (numpy's
                             generator, HYV4FlashEngramLayer.draw_hash_multipliers), the pack's format
  model.safetensors.index.json   the index of the pack's files (NOT the source's)
  experts-LNN.safetensors    one file per MoE layer: every routed expert as MXFP4 planes, the layout Ds41ExpertTier reads
                             layers.<L>.experts.<e>.w1 / .w3  U8 [I, latent/2]   gate / up  (rows of gate_up_proj)
                             layers.<L>.experts.<e>.w2        U8 [latent, I/2]   down
                             <name>_scale                     U8 [rows, cols/32] one E8M0 exponent per 32 elements
  config.json, tokenizer*, chat_template*, generation_config.json, processor_config.json   copied when present

Everything else (attention, shared experts, norms, the n-gram tables, vision, drafter) stays in <source_dir> and is read
in place by the engine. A layer whose file already exists is skipped, so an interrupted run resumes.

4-bit is LOSSY here: the release was not trained for it (unlike DeepSeek-V4.1 / MiMo, which ship MXFP4 experts). Each
layer's signal-to-noise ratio is printed and stored; the engine-side quality check is the plan's phase 3.
"""
import argparse, glob, json, os, shutil, struct, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy4flash_common import mxfp4_quantize, hash_multipliers  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

def shard_index(src):
    """tensor name -> shard path, from the headers (no index file needed; partial downloads list what is complete)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
        for k in h:
            if k != "__metadata__": out[k] = f
    return out

def write_index(dst):
    """model.safetensors.index.json for the pack's files: what the engine's SafetensorsModel opens a directory by."""
    wm = {}
    for f in sorted(glob.glob(os.path.join(dst, "experts-L*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n))
        for k in h:
            if k != "__metadata__": wm[k] = os.path.basename(f)
    if wm: json.dump({"metadata": {"format": "hy4flash-mxfp4-1"}, "weight_map": wm}, open(os.path.join(dst, "model.safetensors.index.json"), "w"))
    return len(wm)

def parse_layers(s, moe_layers):
    if not s: return moe_layers
    want = set()
    for part in s.split(","):
        if "-" in part: a, b = part.split("-"); want.update(range(int(a), int(b) + 1))
        else: want.add(int(part))
    return [L for L in moe_layers if L in want]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--layers", default=""); ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--no-experts", action="store_true")
    a = ap.parse_args()
    src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
    cfg_all = json.load(open(os.path.join(src, "config.json"))); cfg = cfg_all.get("text_config", cfg_all)
    os.makedirs(dst, exist_ok=True)
    idx = shard_index(src)
    nL = cfg["num_hidden_layers"]; E = cfg["n_routed_experts"]; I = cfg["moe_intermediate_size"]
    lat = cfg.get("routed_expert_hidden_size") or cfg["hidden_size"]
    mlp_types = cfg.get("mlp_layer_types") or (["dense"] * cfg.get("first_k_dense_replace", 1) + ["sparse"] * (nL - cfg.get("first_k_dense_replace", 1)))
    moe_layers = [L for L in range(nL) if mlp_types[L] == "sparse"]

    # ---- the sidecar ----
    side = {"format": 1, "source_dir": src, "expert_format": "none" if a.no_experts else "mxfp4",
            "expert_shape": {"n": E, "inter": I, "latent": lat}, "engram_hash_multipliers": {}, "expert_sqnr_db": {}}
    side_path = os.path.join(dst, "hy4flash_ie.json")
    if os.path.exists(side_path):
        old = json.load(open(side_path)); side["expert_sqnr_db"] = old.get("expert_sqnr_db", {})
    for lid in cfg.get("engram_layer_ids") or []:
        name = f"model.layers.{lid - 1}.engram_layer.lookup_table"
        if name not in idx: print(f"  (lookup table of n-gram layer {lid} not in the source yet: multipliers skipped)"); continue
        with safe_open(idx[name], framework="pt") as f: table = f.get_slice(name).get_shape()[0]
        side["engram_hash_multipliers"][str(lid)] = {"table_size": table,
            "multipliers": hash_multipliers(cfg.get("engram_seed", 0), lid, cfg["engram_max_ngram_size"], table)}
    for pat in ("config.json", "generation_config.json", "processor_config.json", "preprocessor_config.json", "tokenizer*", "chat_template*", "special_tokens_map.json", "vocab*"):
        for f in glob.glob(os.path.join(src, pat)):
            if os.path.isfile(f): shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
    json.dump(side, open(side_path, "w"), indent=1)

    # ---- the experts ----
    if not a.no_experts:
        t_all = time.time()
        for L in parse_layers(a.layers, moe_layers):
            out = os.path.join(dst, f"experts-L{L:02d}.safetensors")
            if os.path.exists(out): print(f"layer {L}: exists, skipped"); continue
            gu, dn = f"model.layers.{L}.mlp.experts.gate_up_proj", f"model.layers.{L}.mlp.experts.down_proj"
            if gu not in idx or dn not in idx: print(f"layer {L}: not in the source (yet), skipped"); continue
            t0 = time.time(); tensors = {}; sig = noise = 0.0
            with safe_open(idx[gu], framework="pt") as fg, safe_open(idx[dn], framework="pt") as fd:
                sg, sd = fg.get_slice(gu), fd.get_slice(dn)
                assert sg.get_shape() == [E, 2 * I, lat] and sd.get_shape() == [E, lat, I], (sg.get_shape(), sd.get_shape())
                for e0 in range(0, E, a.chunk):
                    e1 = min(E, e0 + a.chunk)
                    for nm, w in (("gu", sg[e0:e1].float()), ("dn", sd[e0:e1].float())):
                        n, rows, cols = w.shape
                        nib, ex, deq = mxfp4_quantize(w.reshape(n * rows, cols))
                        sig += float((w.double() ** 2).sum()); noise += float(((w.reshape(n * rows, cols) - deq).double() ** 2).sum())
                        nib = nib.reshape(n, rows, cols // 2); ex = ex.reshape(n, rows, cols // 32)
                        for i in range(n):
                            p = f"layers.{L}.experts.{e0 + i}."
                            if nm == "gu":
                                tensors[p + "w1"] = nib[i, :I].clone(); tensors[p + "w1_scale"] = ex[i, :I].clone()
                                tensors[p + "w3"] = nib[i, I:].clone(); tensors[p + "w3_scale"] = ex[i, I:].clone()
                            else:
                                tensors[p + "w2"] = nib[i].clone(); tensors[p + "w2_scale"] = ex[i].clone()
            sqnr = 10.0 * torch.log10(torch.tensor(sig / max(noise, 1e-300))).item() if noise > 0 else float("inf")
            save_file(tensors, out + ".tmp", metadata={"format": "hy4flash-mxfp4-1", "layer": str(L), "sqnr_db": f"{sqnr:.3f}"})
            os.replace(out + ".tmp", out)
            side["expert_sqnr_db"][str(L)] = None if sqnr == float("inf") else round(sqnr, 3)
            json.dump(side, open(side_path, "w"), indent=1)
            print(f"layer {L}: {E} experts -> {os.path.getsize(out) / 2**30:.2f} GiB, SQNR {sqnr:.2f} dB, {time.time() - t0:.0f} s", flush=True)
        print(f"experts done in {(time.time() - t_all) / 60:.1f} min; index lists {write_index(dst)} tensors")
    print("wrote", side_path)

if __name__ == "__main__": main()
