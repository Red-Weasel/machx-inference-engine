#!/usr/bin/env python3
"""Convert the fetched Qwen3.8-Flash-Next MTP head (HF safetensors) into an
engine-native GGUF the existing loader stack can bind.

Transforms (mirroring the main converter's conventions, documented in
scripts/qwen4exp_oracle/README.md):
  - BF16 -> F16 (norm gammas -> F32)
  - zero-centered gammas get +1 folded (hc norms, indexer norms, attn q/k
    norms, the two pre_fc norms, the mixer norm)
  - experts.gate_up_proj [E, 2*ef, H] split -> gate/up [E, ef, H], written as
    GGUF [H, ef, E]; down [E, H, ef] -> GGUF [ef, H, E]
  - 2-D linears keep HF row-major bytes; GGUF shape = (in, out)
  - names -> mtp.* engine-style (attn_q fused with its gate, indexer fused
    qk split into q_proj/k_proj like the main converter)

usage: mtp_to_gguf.py <mtp_head.safetensors> <out.gguf>
"""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/vllm-xpu/lib/python3.12/site-packages"))
try:
    import gguf  # noqa: E402
except ImportError:
    for p in (os.path.expanduser("~/venv/lib/python3.12/site-packages"),):
        sys.path.insert(0, p)
    import gguf  # noqa: E402


def load_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        data0 = 8 + n
        out = {}
        for name, m in hdr.items():
            if name == "__metadata__":
                continue
            f.seek(data0 + m["data_offsets"][0])
            raw = f.read(m["data_offsets"][1] - m["data_offsets"][0])
            assert m["dtype"] == "BF16", (name, m["dtype"])
            u16 = np.frombuffer(raw, dtype=np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            out[name] = f32.reshape(m["shape"]).copy()
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    t = load_st(src)
    print(f"loaded {len(t)} tensors")

    w = gguf.GGUFWriter(dst, arch="qwen4exp-mtp")
    w.add_uint32("qwen4exp-mtp.version", 1)

    def add16(name, a):
        w.add_tensor(name, a.astype(np.float16))

    def add32(name, a):
        w.add_tensor(name, a.astype(np.float32))

    P = "mtp.layers.0."
    # attention (q carries the fused sigmoid gate, same as the backbone)
    add16("mtp.attn_q.weight", t[P + "self_attn.q_proj.weight"])
    add16("mtp.attn_k.weight", t[P + "self_attn.k_proj.weight"])
    add16("mtp.attn_v.weight", t[P + "self_attn.v_proj.weight"])
    add16("mtp.attn_output.weight", t[P + "self_attn.o_proj.weight"])
    add32("mtp.attn_q_norm.weight", t[P + "self_attn.q_norm.weight"] + 1.0)
    add32("mtp.attn_k_norm.weight", t[P + "self_attn.k_norm.weight"] + 1.0)
    # indexer: fused [ (4+1)*128, H ] -> q [512,H], k [128,H]
    qk = t[P + "self_attn.indexer.index_qk_proj.weight"]
    add16("mtp.indexer.q_proj.weight", qk[:512])
    add16("mtp.indexer.k_proj.weight", qk[512:])
    add32("mtp.indexer.q_norm.weight", t[P + "self_attn.indexer.q_layernorm.weight"] + 1.0)
    add32("mtp.indexer.k_norm.weight", t[P + "self_attn.indexer.k_layernorm.weight"] + 1.0)
    # hyper-connections (both sites)
    for hf, ie in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
        add32(f"mtp.{ie}_norm.weight", t[f"{P}{hf}.hc_norm.weight"] + 1.0)
        add16(f"mtp.{ie}_down.weight", t[f"{P}{hf}.input_mix_weight_down.weight"])
        add16(f"mtp.{ie}_up.weight", t[f"{P}{hf}.input_mix_weight_up.weight"])
        add32(f"mtp.{ie}_inject.weight", t[f"{P}{hf}.block_inject_weight.weight"])
    # MoE
    add32("mtp.ffn_gate_inp.weight", t[P + "mlp.gate.weight"])
    gu = t[P + "mlp.experts.gate_up_proj"]          # [E, 2*ef, H]
    ef = gu.shape[1] // 2
    add16("mtp.ffn_gate_exps.weight", gu[:, :ef, :])   # [E, ef, H]
    add16("mtp.ffn_up_exps.weight", gu[:, ef:, :])
    add16("mtp.ffn_down_exps.weight", t[P + "mlp.experts.down_proj"])  # [E, H, ef]
    add16("mtp.ffn_gate_shexp.weight", t[P + "mlp.shared_expert.gate_proj.weight"])
    add16("mtp.ffn_up_shexp.weight", t[P + "mlp.shared_expert.up_proj.weight"])
    add16("mtp.ffn_down_shexp.weight", t[P + "mlp.shared_expert.down_proj.weight"])
    add32("mtp.ffn_gate_inp_shexp.weight",
          t[P + "mlp.shared_expert_gate.weight"].reshape(-1))
    # fusion front-end + final mixer
    add32("mtp.pre_fc_norm_hidden.weight", t["mtp.pre_fc_norm_hidden.weight"] + 1.0)
    add32("mtp.pre_fc_norm_embedding.weight", t["mtp.pre_fc_norm_embedding.weight"] + 1.0)
    add16("mtp.fc_hidden.weight", t["mtp.fc_hidden.weight"])
    add16("mtp.fc_embedding.weight", t["mtp.fc_embedding.weight"])
    add32("mtp.out_hc_norm.weight", t["mtp.hyper_connection_mixer.hc_norm.weight"] + 1.0)
    add16("mtp.out_hc_down.weight", t["mtp.hyper_connection_mixer.input_mix_weight_down.weight"])
    add16("mtp.out_hc_up.weight", t["mtp.hyper_connection_mixer.input_mix_weight_up.weight"])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"WROTE {dst}")


if __name__ == "__main__":
    main()
