"""GGUF -> HF weight loader for single Qwen3.8-Flash-Next (qwen4exp) blocks.

Loads and dequantizes every tensor of ONE chosen block from the 4-shard
UD-Q4_K_XL GGUF to fp32 torch tensors, INVERTING the llama.cpp converter
transforms so the result drops into the HF module's state_dict exactly.

Converter transforms inverted here (sources: llamacpp-pr27742.diff conversion/
qwen4exp.py + the inherited classes read from ~/llama.cpp/conversion/
qwen.py: Qwen3NextModel.modify_tensors and _LinearAttentionVReorderBase):

  I1  zero-centered RMSNorm gammas: converter stores gamma_gguf = 1 + w_hf for every
      name ending "norm.weight" except linear_attn.norm.weight (Qwen3NextModel rule),
      plus the explicit qwen4exp additions (indexer q/k layernorm, ple norm_key/
      query/conv).  Inversion: w_hf = gamma_gguf - 1 for attn_q_norm, attn_k_norm,
      indexer.q_norm, indexer.k_norm, hc_attn_norm, hc_ffn_norm, output_hc_norm,
      ple_norm_key, ple_norm_query, ple_norm_conv.
      NOT inverted (stored raw): ssm_norm (HF RMSNormGated weight, ones-init).
  I2  ssm_a = -exp(A_log)  ->  A_log = log(-ssm_a)   (all 48 stored values verified
      negative in the GGUF).
  I3  dt_bias was renamed to ssm_dt.bias -> renamed back.
  I4  DeltaNet V-head reorder (_LinearAttentionVReorderBase): HF stores V heads
      grouped by K head [K0:v0,v1,v2, K1:v0,v1,v2, ...]; the converter re-orders to
      tiled [r0:K0..K15, r1:K0..K15, r2:K0..K15], i.e. gguf_head[r*16+k] =
      hf_head[k*3+r].  Inversion = the same reshape/transpose with (3,16) instead of
      (16,3).  Applied to: attn_qkv V rows (rows 4096:10240), attn_gate rows,
      ssm_alpha rows, ssm_beta rows, ssm_a, ssm_dt.bias, ssm_conv1d V channels
      (4096:10240), ssm_out COLUMNS.
  I5  conv1d kernels stored squeezed [channels, k] -> unsqueeze(1) back to the HF
      Conv1d [channels, 1, k] (both ssm_conv1d and ple_conv1d).
  I6  indexer.index_qk_proj was split into indexer.q_proj (rows 0:512) and
      indexer.k_proj (rows 512:640) -> concatenated back.
  I7  experts gate_up_proj was split into ffn_gate_exps (cols 0:640 of the fused
      out-dim) and ffn_up_exps (cols 640:1280) -> concatenated back on dim 1.
  I8  shared_expert_gate [1, 2560] was squeezed to ffn_gate_inp_shexp [2560]
      -> reshaped back to [1, 2560].

NOT transforms (orientation only): gguf-py's dequantize() already returns numpy
arrays in HF (out_features, in_features) row-major order, so plain 2D projections
(attn_q/k/v/output, ple_key/value, hc down/up/inject, shexp mlp, router) load
directly; ffn_down_exps dequantizes to (512, 2560, 640) = HF down_proj directly.

The 26.8 GiB IQ4_NL PLE table is NEVER materialized: rows are gathered and
dequantized on demand (ple_row_reader), 90 bytes -> 160 fp32 per row.

CLI:  python gguf_weights.py --layer L [--summary]
      prints every produced HF key with shape and fp64 sum (checksums).
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import qwen4exp_ref as ref

GGUF_GLOB = os.path.expanduser("~/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/*.gguf")

NUM_K_HEADS = 16
NUM_V_HEADS = 48
V_PER_K = 3
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM      # 2048
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM    # 6144


def _swap_heads(t, dim, a, b, head_dim):
    """Reshape dim into [a, b, head_dim], swap (a, b), flatten back."""
    shape = list(t.shape)
    d = dim if dim >= 0 else dim + len(shape)
    ns = shape[:d] + [a, b, head_dim] + shape[d + 1:]
    t = t.reshape(*ns)
    perm = list(range(len(ns)))
    perm[d], perm[d + 1] = perm[d + 1], perm[d]
    return t.permute(*perm).contiguous().reshape(*shape)


def tiled_to_grouped(t, dim, head_dim):
    """Invert the converter's grouped->tiled V-head reorder (I4)."""
    return _swap_heads(t, dim, V_PER_K, NUM_K_HEADS, head_dim)


def grouped_to_tiled(t, dim, head_dim):
    """The converter's forward reorder (used only by the round-trip self-test)."""
    return _swap_heads(t, dim, NUM_K_HEADS, V_PER_K, head_dim)


class GgufStore:
    """Name -> tensor index over all shards, with fp32 dequant."""

    def __init__(self, pattern=GGUF_GLOB):
        gguf = ref.import_gguf()
        from gguf import GGUFReader

        import importlib

        self._gq = importlib.import_module("gguf.quants")
        self._QT = gguf.GGMLQuantizationType
        self.shards = sorted(glob.glob(pattern))
        if not self.shards:
            raise FileNotFoundError(f"no GGUF shards match {pattern}")
        self.readers = [GGUFReader(s) for s in self.shards]
        self.tensors = {}
        for r in self.readers:
            for t in r.tensors:
                self.tensors[t.name] = t
        self.fields = {}
        for r in self.readers:
            for f in r.fields.values():
                if f.name not in self.fields:
                    self.fields[f.name] = f

    def kv(self, name):
        f = self.fields[name]
        return f.contents()

    def raw(self, name):
        return self.tensors[name]

    def dequant_np(self, name):
        t = self.tensors[name]
        if t.tensor_type == self._QT.F32:
            a = np.asarray(t.data)
            if a.dtype != np.float32:
                a = a.view(np.float32)
            return np.array(a, dtype=np.float32)  # copy: memmap views are read-only
        if t.tensor_type == self._QT.F16:
            return np.array(np.asarray(t.data), dtype=np.float32)
        return np.ascontiguousarray(
            self._gq.dequantize(t.data, t.tensor_type), dtype=np.float32
        )

    def dequant(self, name):
        import torch

        return torch.from_numpy(self.dequant_np(name))

    def ple_row_reader(self):
        """Callable ids[int64 np array] -> fp32 np [n, 160]; row-sparse IQ4_NL dequant."""
        t = self.tensors["per_layer_token_embd.weight"]
        qt = t.tensor_type
        gq = self._gq
        data = t.data  # memmap (320001536, 90) uint8

        def read(ids_np):
            rows = np.ascontiguousarray(data[ids_np])
            return np.ascontiguousarray(gq.dequantize(rows, qt), dtype=np.float32)

        return read


def _selftest_reorder():
    """Round-trip: tiled_to_grouped inverts the converter's exact reorder."""
    import torch

    x = torch.arange(NUM_V_HEADS * HEAD_V_DIM, dtype=torch.float32).unsqueeze(-1)
    fwd = grouped_to_tiled(x, 0, HEAD_V_DIM)
    back = tiled_to_grouped(fwd, 0, HEAD_V_DIM)
    assert torch.equal(back, x), "V-head reorder round-trip failed"
    # spot-check the mapping law gguf[r*16+k] == hf[k*3+r]
    heads = torch.arange(NUM_V_HEADS, dtype=torch.float32).unsqueeze(-1)
    tiled = grouped_to_tiled(heads, 0, 1).squeeze(-1)
    for r in range(V_PER_K):
        for k in range(NUM_K_HEADS):
            assert int(tiled[r * NUM_K_HEADS + k]) == k * V_PER_K + r


def is_linear_attention(layer):
    return layer % 4 != 3


def block_state_dict(store, layer):
    """(state_dict, notes) for Qwen4ExpTextDecoderLayer `layer`, HF names/orient."""
    import torch

    _selftest_reorder()
    B = f"blk.{layer}."
    sd = {}
    notes = []

    def deq(suffix):
        return store.dequant(B + suffix)

    # ---- token mixer -------------------------------------------------------
    if is_linear_attention(layer):
        qkv = deq("attn_qkv.weight")                      # [10240, 2560] rows q|k|v_tiled
        v = tiled_to_grouped(qkv[KEY_DIM * 2:], 0, HEAD_V_DIM)
        sd["linear_attn.in_proj_qkv.weight"] = torch.cat([qkv[: KEY_DIM * 2], v], dim=0)
        notes.append("I4 attn_qkv: V rows un-tiled -> in_proj_qkv")

        sd["linear_attn.in_proj_z.weight"] = tiled_to_grouped(deq("attn_gate.weight"), 0, HEAD_V_DIM)
        sd["linear_attn.in_proj_a.weight"] = tiled_to_grouped(deq("ssm_alpha.weight"), 0, 1)
        sd["linear_attn.in_proj_b.weight"] = tiled_to_grouped(deq("ssm_beta.weight"), 0, 1)
        notes.append("I4 attn_gate/ssm_alpha/ssm_beta rows un-tiled -> in_proj_z/a/b")

        ssm_a = deq("ssm_a")                              # [48], = -exp(A_log), tiled
        assert bool((ssm_a < 0).all()), "ssm_a must be strictly negative (= -exp(A_log))"
        sd["linear_attn.A_log"] = tiled_to_grouped(
            torch.log(-ssm_a).unsqueeze(-1), 0, 1
        ).squeeze(-1)
        notes.append("I2+I4 ssm_a: A_log = log(-ssm_a), un-tiled")

        sd["linear_attn.dt_bias"] = tiled_to_grouped(
            deq("ssm_dt.bias").unsqueeze(-1), 0, 1
        ).squeeze(-1)
        notes.append("I3+I4 ssm_dt.bias -> dt_bias, un-tiled")

        conv = deq("ssm_conv1d.weight")                   # [10240, 4] channels q|k|v_tiled
        vch = tiled_to_grouped(conv[KEY_DIM * 2:], 0, HEAD_V_DIM)
        sd["linear_attn.conv1d.weight"] = torch.cat([conv[: KEY_DIM * 2], vch], dim=0).unsqueeze(1)
        notes.append("I4+I5 ssm_conv1d: V channels un-tiled, unsqueeze -> [10240,1,4]")

        sd["linear_attn.out_proj.weight"] = tiled_to_grouped(deq("ssm_out.weight"), 1, HEAD_V_DIM)
        notes.append("I4 ssm_out: COLUMNS un-tiled -> out_proj")

        sd["linear_attn.norm.weight"] = deq("ssm_norm.weight")
        notes.append("ssm_norm loaded RAW (RMSNormGated weight; no +1 in converter)")
    else:
        sd["self_attn.q_proj.weight"] = deq("attn_q.weight")     # [12288, 2560] q|gate per head
        sd["self_attn.k_proj.weight"] = deq("attn_k.weight")
        sd["self_attn.v_proj.weight"] = deq("attn_v.weight")
        sd["self_attn.o_proj.weight"] = deq("attn_output.weight")
        sd["self_attn.q_norm.weight"] = deq("attn_q_norm.weight") - 1.0
        sd["self_attn.k_norm.weight"] = deq("attn_k_norm.weight") - 1.0
        notes.append("I1 attn_q_norm/attn_k_norm: gamma-1 (zero-centered HF RMSNorm)")

        qi = deq("indexer.q_proj.weight")                        # [512, 2560]
        ki = deq("indexer.k_proj.weight")                        # [128, 2560]
        sd["self_attn.indexer.index_qk_proj.weight"] = torch.cat([qi, ki], dim=0)
        notes.append("I6 indexer q_proj|k_proj re-fused -> index_qk_proj [640, 2560]")
        sd["self_attn.indexer.q_layernorm.weight"] = deq("indexer.q_norm.weight") - 1.0
        sd["self_attn.indexer.k_layernorm.weight"] = deq("indexer.k_norm.weight") - 1.0
        notes.append("I1 indexer q/k_norm: gamma-1")

    # ---- hyper-connections -------------------------------------------------
    for hf, gg in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
        sd[f"{hf}.hc_norm.weight"] = deq(f"{gg}_norm.weight") - 1.0
        sd[f"{hf}.input_mix_weight_down.weight"] = deq(f"{gg}_down.weight")   # [320, 10240]
        sd[f"{hf}.input_mix_weight_up.weight"] = deq(f"{gg}_up.weight")       # [10240, 320]
        sd[f"{hf}.block_inject_weight.weight"] = deq(f"{gg}_inject.weight")   # [4, 10240]
    notes.append("I1 hc_attn_norm/hc_ffn_norm: gamma-1")

    # ---- MoE ---------------------------------------------------------------
    sd["mlp.gate.weight"] = deq("ffn_gate_inp.weight")                        # [512, 2560]
    gate = deq("ffn_gate_exps.weight")                                        # [512, 640, 2560]
    up = deq("ffn_up_exps.weight")                                            # [512, 640, 2560]
    sd["mlp.experts.gate_up_proj"] = torch.cat([gate, up], dim=1)             # [512, 1280, 2560]
    notes.append("I7 ffn_gate_exps|ffn_up_exps re-fused -> experts.gate_up_proj")
    sd["mlp.experts.down_proj"] = deq("ffn_down_exps.weight")                 # [512, 2560, 640]
    sd["mlp.shared_expert.gate_proj.weight"] = deq("ffn_gate_shexp.weight")
    sd["mlp.shared_expert.up_proj.weight"] = deq("ffn_up_shexp.weight")
    sd["mlp.shared_expert.down_proj.weight"] = deq("ffn_down_shexp.weight")
    sd["mlp.shared_expert_gate.weight"] = deq("ffn_gate_inp_shexp.weight").reshape(1, -1)
    notes.append("I8 ffn_gate_inp_shexp [2560] -> shared_expert_gate [1, 2560]")

    # ---- PLE (layer 1 only) ------------------------------------------------
    if layer == 1:
        sd["ple.key_proj.weight"] = deq("ple_key.weight")                     # [10240, 2560]
        sd["ple.value_proj.weight"] = deq("ple_value.weight")                 # [2560, 2560]
        sd["ple.norm_key.weight"] = deq("ple_norm_key.weight") - 1.0
        sd["ple.norm_query.weight"] = deq("ple_norm_query.weight") - 1.0
        sd["ple.norm_conv.weight"] = deq("ple_norm_conv.weight") - 1.0
        notes.append("I1 ple_norm_key/query/conv: gamma-1")
        sd["ple.conv1d.weight"] = deq("ple_conv1d.weight").unsqueeze(1)       # [10240, 1, 4]
        notes.append("I5 ple_conv1d unsqueeze -> [10240, 1, 4] (no V reorder: not linear_attn)")

    return sd, notes


def check_ple_constants(store, ngram_module):
    """Assert the HF module's hash buffers equal the GGUF metadata exactly."""
    mult = [int(x) for x in store.kv("qwen4exp.ple.layer_multipliers")]
    offs = [int(x) for x in store.kv("qwen4exp.ple.head_offsets")]
    vocs = [int(x) for x in store.kv("qwen4exp.ple.head_vocab_sizes")]
    got_m = [int(x) for x in ngram_module.layer_multipliers.tolist()]
    got_o = [int(x) for x in ngram_module.ngram_heads_offsets.tolist()]
    got_v = [int(x) for x in ngram_module.ngram_heads_vocab_sizes.tolist()]
    assert got_m == mult, f"layer_multipliers mismatch: module {got_m} vs gguf {mult}"
    assert got_o == offs, "head_offsets mismatch vs gguf metadata"
    assert got_v == vocs, "head_vocab_sizes mismatch vs gguf metadata"
    eos = int(store.kv("qwen4exp.ple.eos_token_id"))
    assert ngram_module.eos_token_id == eos, f"eos mismatch {ngram_module.eos_token_id} vs {eos}"
    return {"layer_multipliers": mult, "head_offsets": offs, "head_vocab_sizes": vocs, "eos": eos}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--summary", action="store_true", help="print per-tensor checksums")
    args = ap.parse_args()

    store = GgufStore()
    sd, notes = block_state_dict(store, args.layer)
    kind = "linear_attention" if is_linear_attention(args.layer) else "full_attention"
    print(f"layer {args.layer} ({kind}): {len(sd)} HF tensors")
    for n in notes:
        print("  [inversion]", n)
    if args.summary:
        for k in sorted(sd):
            v = sd[k]
            print(f"  {k:48s} {str(list(v.shape)):22s} sum={float(v.double().sum()):+.6e}")


if __name__ == "__main__":
    main()
