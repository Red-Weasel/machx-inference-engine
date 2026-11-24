"""Frozen experts kept in their checkpoint form (NVFP4 / MXFP8) and decoded on the fly: the piece that lets the whole
model sit on ONE 32 GiB card for training.

Why: the reference decodes all 48 expert tensors to bf16 at load (about 49 GiB + 3.6 GiB dense = 53 GiB), which fits
neither card. The experts are frozen, so they never need to exist as trainable tensors: keep the 14.6 GiB of codes on
the device, decode only the experts a forward call hits, use them, drop them. The backward pass needs the decoded
matrices again (d output / d input of a linear is its weight), so each chunk of experts runs under
torch.utils.checkpoint: the first run keeps no graph, the backward decodes that chunk once more and frees it.
Peak extra memory = one chunk decoded (16 experts x 15.2 MiB = 0.24 GiB in bf16) plus the decode temporaries.

Decoding follows the reference converter (transformers core_model_loading.Inkling2LoopDequantizeExperts + the row
de-interleave of w13) and is bit-identical to its loaded gate_up_proj / down_proj with amax="reference"
(check_compressed_experts.py). amax="exact" uses the checkpoint's fp32 amax instead of the reference loader's
bf16-rounded one -- that is what the owner's engine does (SPEC section 9), so it is the right setting for a model that
will be served by the engine; the two differ by a per-expert constant factor within +-0.4 %.
"""
import json
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def e4m3_table():
    """float8_e4m3fn -> fp32 for all 256 bit patterns (1 sign, 4 exponent bias 7, 3 mantissa; subnormals at e = 0)."""
    return torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()


class SafetensorsReader:
    """Raw tensor access by name (numpy memmap), no dtype conversion."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self.header_len = struct.unpack("<Q", f.read(8))[0]
            self.header_bytes = f.read(self.header_len)
        self.header = json.loads(self.header_bytes)
        self.data_start = 8 + self.header_len
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def raw(self, name):
        a, b = self.header[name]["data_offsets"]
        return self.mm[self.data_start + a:self.data_start + b]

    def u8(self, name):          # U8 and F8_E4M3 tensors as their bytes
        return torch.from_numpy(np.array(self.raw(name))).view(*self.header[name]["shape"])

    def f32(self, name):
        return torch.from_numpy(np.array(self.raw(name)).view(np.float32).copy()).view(*self.header[name]["shape"])


COMPILE = True
DETERMINISTIC = True       # see _chunk: the reference's index_add uses atomic adds on the card
_DECODE = None
ROW_BUDGET = 16384          # padded token rows per expert chunk (x, gate/up, y: 75 MB each in bf16)
_ACT = None                 # the compiled SwiGLU (the caps are config-wide: one for every layer)


def _compiled_decode():
    global _DECODE
    if _DECODE is None:
        _DECODE = torch.compile(CompressedExperts.decode, dynamic=False)     # as measured in decode_bench.py
    return _DECODE


class CompressedExperts(nn.Module):
    """Drop-in for Inkling2LoopExperts (layers[i].mlp.experts): same forward signature, same arithmetic and rounding
    points as its per-expert loop (modeling_inkling2_loop.py:787-811), weights decoded per call."""

    def __init__(self, reader: SafetensorsReader, layer: int, act_fn, amax="reference", dtype=torch.bfloat16,
                 use_checkpoint=True, experts_per_chunk=32, batched=True):
        super().__init__()
        p = f"model.llm.layers.{layer}.mlp.experts."
        self.nvfp4 = (p + "w13_weight_amax") in reader.header
        self.num_experts, self.act_fn, self.dtype = 128, act_fn, dtype
        self.use_checkpoint, self.chunk, self.batched = use_checkpoint, experts_per_chunk, batched
        for short in ("w13", "w2"):
            self.register_buffer(short + "_codes", reader.u8(p + short + "_weight"), persistent=False)
            self.register_buffer(short + "_scale", reader.u8(p + short + "_weight_scale"), persistent=False)
            if self.nvfp4:
                am = reader.f32(p + short + "_weight_amax")
                den = reader.f32(p + short + "_weight_outer_scale_denominator")
                if amax == "reference":      # the reference loader rounds both to bf16 before dividing
                    am, den = am.to(torch.bfloat16).float(), den.to(torch.bfloat16).float()
                self.register_buffer(short + "_factor", am / den, persistent=False)
        e2m1 = torch.tensor(E2M1)
        byte = torch.arange(256)
        self.register_buffer("byte_to_values", torch.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), dim=-1), persistent=False)
        self.register_buffer("e4m3", e4m3_table(), persistent=False)

    def compressed_bytes(self):
        return sum(b.numel() * b.element_size() for b in self.buffers())

    @torch.no_grad()
    def decode(self, short, idx):
        """Decoded matrices of experts `idx` (LongTensor [n]) -> [n, rows, columns] in self.dtype.
        w13 comes back de-interleaved: rows 0..1151 = gate, 1152..2303 = up (the reference's Interleave(dim=1))."""
        codes = getattr(self, short + "_codes")[idx]
        scale = getattr(self, short + "_scale")[idx]
        n, rows = codes.shape[:2]
        if self.nvfp4:
            block = self.e4m3[scale.long()]                                             # [n, rows, cols / 16]
            vals = self.byte_to_values[codes.long()].view(*block.shape, 16) * block[..., None]
            vals = vals * getattr(self, short + "_factor")[idx, None, None, None]
        else:
            vals = self.e4m3[codes.long()].view(*scale.shape, 32)
            vals = vals * torch.exp2(scale.float() - 127)[..., None]
        w = vals.reshape(n, rows, -1).to(self.dtype)
        if short == "w13":
            w = w.view(n, rows // 2, 2, -1).transpose(1, 2).reshape(n, rows, -1)
        return w

    def decode_fast(self, short, idx):
        """decode() through torch.compile: one fused kernel instead of ~10 passes over fp32 temporaries (15x faster on
        the B70, bit-identical: decode_bench.py). The index is padded to the chunk size so the compiled graph has one
        shape per (format, tensor)."""
        if not COMPILE or idx.device.type == "cpu":
            return self.decode(short, idx)
        n = idx.numel()
        size = 1 << (n - 1).bit_length()                 # 1, 2, 4, ... 32: a handful of compiled shapes
        if n < size:
            idx = torch.cat([idx, idx[-1:].expand(size - n)])
        return _compiled_decode()(self, short, idx)[:n]

    def _act(self, gate, up):
        """The softcapped SwiGLU through torch.compile on the card: one fused kernel each way (8x faster) that keeps only
        its two bf16 inputs for the backward -- eager keeps fp32 copies and tanh outputs, 5.3 GiB at 4,096 tokens, the
        largest item of the backward's peak. Not bit-identical to eager (tanh / sigmoid differ by an ulp at most)."""
        if not COMPILE or gate.device.type == "cpu":
            return self.act_fn(gate, up)
        global _ACT
        if _ACT is None:
            _ACT = torch.compile(self.act_fn, dynamic=True)
        return _ACT(gate, up)

    def _chunk(self, hidden_states, top_k_index, top_k_weights, idx):
        """Contribution of the experts `idx` to every token, fp32 [tokens, hidden]."""
        final = torch.zeros_like(hidden_states, dtype=torch.float32)
        w13, w2 = self.decode_fast("w13", idx), self.decode_fast("w2", idx)
        if not self.batched:                       # the reference's per-expert loop, one small matmul pair per expert
            for j, e in enumerate(idx):
                token_idx, top_k_pos = torch.where(top_k_index == e)
                gate, up = F.linear(hidden_states[token_idx], w13[j]).chunk(2, dim=-1)
                y = F.linear(self.act_fn(gate, up), w2[j])
                y = y * top_k_weights[token_idx, top_k_pos, None]
                final = final.index_add(0, token_idx, y.to(final.dtype))
            return final
        # Batched: the chunk's experts in two bmm calls. Each expert's tokens are packed into a row of a
        # [experts, n_max, hidden] block (zero-padded), so the number of device kernels per chunk is constant
        # instead of ~10 per expert -- on the B70 the loop is launch-bound (0.4 s per MoE call at any length).
        tok, kpos, j = (top_k_index[..., None] == idx).nonzero(as_tuple=True)
        order = torch.argsort(j, stable=True)
        tok, kpos, j = tok[order], kpos[order], j[order]
        counts = torch.bincount(j, minlength=idx.numel())
        pos = torch.arange(j.numel(), device=j.device) - (torch.cumsum(counts, 0) - counts)[j]
        x = hidden_states.new_zeros(idx.numel(), int(counts.max()), hidden_states.shape[-1])
        x = x.index_put((j, pos), hidden_states[tok])
        gate, up = torch.bmm(x, w13.transpose(1, 2)).chunk(2, dim=-1)
        y = torch.bmm(self._act(gate, up), w2.transpose(1, 2))[j, pos]
        y = y * top_k_weights[tok, kpos, None]
        if DETERMINISTIC:      # each (token, k) slot written once: no atomic float adds, the same sum on every run
            # per k a token appears at most once, so each k is a plain (non-accumulating) write; k = 0..7 summed in
            # order, chunks added in order
            y = y.to(torch.float32)
            out = torch.zeros_like(final)
            for k in range(top_k_index.shape[1]):
                sel = (kpos == k).nonzero(as_tuple=True)[0]
                out = out + torch.zeros_like(final).index_put((tok[sel],), y[sel])
            return out
        return final.index_add(0, tok, y.to(final.dtype))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # One checkpoint per chunk of experts: the forward keeps no decoded matrix, the backward re-decodes one chunk
        # at a time, so the decoded weights alive at any moment are `chunk` experts (16 x 15.2 MiB), not the layer.
        ckpt = self.use_checkpoint and torch.is_grad_enabled() and hidden_states.requires_grad
        final = torch.zeros_like(hidden_states, dtype=torch.float32)
        # chunks of experts with similar token counts: the bmm pads every expert of a chunk to the busiest one, so
        # mixing a busy expert with idle ones wastes compute and memory (the order is deterministic: stable sort)
        # and at most ROW_BUDGET padded rows a chunk: at long examples one expert can take most tokens
        counts = torch.bincount(top_k_index.flatten(), minlength=self.num_experts)
        hit = torch.argsort(counts, descending=True, stable=True)[:int((counts > 0).sum())]
        cnt = counts[hit].tolist()
        plan, i = [], 0
        while i < len(cnt):
            k = max(1, min(self.chunk, ROW_BUDGET // max(1, cnt[i])))
            plan.append((i, min(len(cnt), i + k)))
            i += k
        for s, e in plan:
            idx = hit[s:e]
            if ckpt:
                final = final + checkpoint(self._chunk, hidden_states, top_k_index, top_k_weights, idx, use_reentrant=False)
            else:
                final = final + self._chunk(hidden_states, top_k_index, top_k_weights, idx)
        return final.to(hidden_states.dtype)


def compress_model_experts(model, safetensors_path, layers=None, amax="reference", device=None):
    """Replace mlp.experts of the given layers (default all) by CompressedExperts read from the checkpoint.
    Frees the decoded bf16 copies. Returns the reader."""
    reader = SafetensorsReader(safetensors_path)
    L = model.model.language_model.layers
    for i in (range(len(L)) if layers is None else layers):
        old = L[i].mlp.experts
        new = CompressedExperts(reader, i, old.act_fn, amax=amax)
        L[i].mlp.experts = new.to(device) if device is not None else new
        del old
    return reader


@torch.no_grad()
def set_reference_experts_amax(model, safetensors_path, amax="exact", layers=None, chunk=16):
    """Re-decode the NVFP4 experts of the REFERENCE expert modules (the bf16 gate_up_proj / down_proj the reference
    loader filled) in place with the chosen amax, so the CPU trainer gets the engine's numbers at no per-step cost.

    amax="exact"     : fp32 amax / fp32 denominator, as the owner's engine computes the per-expert factor (SPEC 9).
    amax="reference" : the reference loader's bf16-rounded pair (re-decoding with it reproduces the loaded tensors
                       bit for bit -- check_compressed_experts.py -- so this is a no-op apart from the time).
    MXFP8 layers have no amax and are left untouched. Returns {"layers": [...], "elements_changed": n, "elements": N}.
    About 4 s per NVFP4 layer on the CPU."""
    reader = SafetensorsReader(safetensors_path)
    L = model.model.language_model.layers
    out = {"layers": [], "elements_changed": 0, "elements": 0}
    for i in (range(len(L)) if layers is None else layers):
        ex = L[i].mlp.experts
        ce = CompressedExperts(reader, i, ex.act_fn, amax=amax)
        if not ce.nvfp4:
            continue
        for short, param in (("w13", ex.gate_up_proj), ("w2", ex.down_proj)):
            for s in range(0, ce.num_experts, chunk):
                idx = torch.arange(s, min(s + chunk, ce.num_experts))
                new = ce.decode(short, idx).to(param.dtype)
                out["elements_changed"] += int((new.view(torch.int16) != param.data[idx].view(torch.int16)).sum())
                out["elements"] += new.numel()
                param.data[idx] = new
        out["layers"].append(i)
        del ce
    return out
