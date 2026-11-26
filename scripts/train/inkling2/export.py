"""Write a trained model folder: the source folder's files, the weight file patched with the adapter.

The weight file is copied whole (header byte for byte), then only the 96 adapted attention tensors are overwritten in
place with bf16(fp32(W) + scale * B @ A) -- the expression and op order of model.LoRALinear.merged_weight, so the
folder holds exactly the model that was trained. The expert codes and every other tensor are the original bytes.
Other files: hard links on the same drive, copies otherwise (the folder stands alone: no symlinks); the run-time
ablation / steering files (ie_*.json, *.f32) are always copied, so the engine applies the same ablation.
The folder is written under a temporary name and renamed at the end: it exists only when complete.
"""
import json
import os
import shutil
import time

import numpy as np
import torch

from .compressed_experts import SafetensorsReader
from .model import TARGETS, WEIGHTS


def _is_ablation_file(name):
    return name.startswith("ie_") or name.endswith(".f32")


def export(base, out, adapter_tensors, meta, training_md, log=print):
    """adapter_tensors: {"<layer>.<t>.A"/".B": cpu fp32}; meta: {"scale", "keys"}. Returns the export info."""
    if os.path.exists(out):
        raise FileExistsError(f"{out} exists")
    tmp = out + ".writing"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    t0 = time.time()
    parent = os.path.dirname(os.path.abspath(out))
    same_fs = os.stat(parent).st_dev == os.stat(base).st_dev
    link = os.link if same_fs else shutil.copy2
    for name in sorted(os.listdir(base)):
        s, d = os.path.join(base, name), os.path.join(tmp, name)
        if name in (WEIGHTS, ".cache") or name.endswith(".writing") or name == "TRAINING.md":
            continue
        if os.path.isdir(s):
            shutil.copytree(s, d, copy_function=link)
        elif _is_ablation_file(name):
            shutil.copy2(s, d)
        else:
            link(s, d)
    src = SafetensorsReader(os.path.join(base, WEIGHTS))
    dst = os.path.join(tmp, WEIGHTS)
    shutil.copyfile(os.path.join(base, WEIGHTS), dst)
    log(f"export: weight file copied in {time.time() - t0:.0f} s")
    changed = patch_attention(src, dst, adapter_tensors, meta)
    info = {"base": os.path.abspath(base), "tensors_patched": len(changed),
            "elements_changed": sum(c["elements_changed"] for c in changed),
            "elements_in_patched_tensors": sum(c["elements"] for c in changed), "changed": changed}
    json.dump(info, open(os.path.join(tmp, "EXPORT_INFO.json"), "w"), indent=1)
    with open(os.path.join(tmp, "TRAINING.md"), "w") as f:
        f.write(training_md)
    os.rename(tmp, out)
    log(f"export: {len(changed)} tensors patched, folder written in {time.time() - t0:.0f} s")
    return info


def patch_attention(src, dst, adapter_tensors, meta):
    """Overwrite the adapted attention tensors of the weight file `dst` (a byte copy of src's file) in place with
    bf16(fp32(W) + scale * B @ A), W read from `src` (a SafetensorsReader of the base weights). Returns the change list."""
    changed = []
    with open(dst, "r+b") as f:
        for key in meta["keys"]:
            layer, t = key.split(".")
            name = f"model.llm.layers.{layer}.attn.{TARGETS[t][1]}"
            h = src.header[name]
            assert h["dtype"] == "BF16", name
            A, B = adapter_tensors[key + ".A"], adapter_tensors[key + ".B"]
            W = torch.from_numpy(np.array(src.raw(name)).view(np.int16).copy()).view(torch.bfloat16).view(*h["shape"])
            Wn = (W.float() + meta["scale"] * (B @ A)).to(torch.bfloat16)
            new = Wn.contiguous().view(torch.int16).numpy().tobytes()
            assert len(new) == h["data_offsets"][1] - h["data_offsets"][0]
            f.seek(src.data_start + h["data_offsets"][0])
            f.write(new)
            changed.append({"tensor": name, "elements_changed": int((Wn.view(torch.int16) != W.view(torch.int16)).sum()),
                            "elements": W.numel()})
        f.flush()
        os.fsync(f.fileno())
    return changed
