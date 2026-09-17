"""Op-parity oracle: golden tensors for isolated qwen4exp ops, SYNTHETIC weights.

    python run_op_oracle.py --out-dir DIR [--seed S]

Runs the REAL HF reference modules from modeling_qwen4_exp.py (never re-implemented
math) on tiny shapes with fixed-seed fp32 randn weights, and dumps one npz per op.
These seed the C++ per-op parity tests without any GGUF I/O.

Ops (npz files, keys documented in README.md):
  hc.npz            Qwen4ExpTextGatedResidual mix + the decoder layer's combine law
                    (tiny: H=32, hc=4, lowrank=8)
  deltanet_step.npz torch_recurrent_gated_delta_rule single-step recurrence unrolled
                    over T steps, with a nonzero initial state, plus the chunked
                    prefill rule (chunk_size 4, T=8 -> two chunks) on the same data
  ple.npz           Qwen4ExpTextPLELayer full path (hash -> embed -> key/value ->
                    gate -> dilated conv), tiny config with a real (small) table
  moe.npz           Qwen4ExpTextSparseMoeBlock router + experts + shared expert
                    (tiny: H=32, E=8, top-3, ffn 16)

Norm weights are dumped in the HF convention: zero-centered w for
Qwen4ExpTextRMSNorm (applied as 1+w; the GGUF convention pre-folds the +1), RAW
weight for RMSNormGated.  The C++ side must apply the same convention it tests.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import qwen4exp_ref as ref


def _fill_params(module, gen, torch, scale=0.5, norm_scale=0.1):
    """Deterministically randomize every parameter (sorted name order)."""
    with torch.no_grad():
        for name, p in sorted(module.named_parameters()):
            r = torch.randn(p.shape, generator=gen, dtype=torch.float32)
            if name.endswith("norm.weight") or ".hc_norm." in name or "norm_" in name:
                p.copy_(r * norm_scale)  # zero-centered / near-init norm gammas
            else:
                p.copy_(r * scale / max(1.0, p.shape[-1] ** 0.5))


def _np(t):
    return t.detach().to("cpu").to(__import__("torch").float32).numpy()


def dump_hc(mod, out_dir, seed):
    import torch

    gen = torch.Generator().manual_seed(seed)
    H, HC, R, T = 32, 4, 8, 5
    cfg = ref.Qwen4ExpTextConfig(hc_count=HC, hidden_size=H, hc_lowrank=R, rms_norm_eps=1e-6)
    m = mod.Qwen4ExpTextGatedResidual(cfg, use_combine=True).float().eval()
    _fill_params(m, gen, torch)
    x = torch.randn(1, T, HC * H, generator=gen)
    block_out = torch.randn(1, T, H, generator=gen)  # synthetic token-mixer output
    with torch.no_grad():
        mixed, hyper_input, inj_w = m(x)
        injection = block_out.unsqueeze(-2) * inj_w.unsqueeze(-1)  # decoder-layer combine law
        combined = hyper_input + injection.flatten(-2)
        m_nc = mod.Qwen4ExpTextGatedResidual(cfg, use_combine=False).float().eval()
        m_nc.load_state_dict({k: v for k, v in m.state_dict().items()
                              if not k.startswith("block_inject_weight")})
        mixed_nc = m_nc(x)
        assert torch.equal(mixed_nc, mixed)  # head mixer == mix with inject skipped
    np.savez(
        os.path.join(out_dir, "hc.npz"),
        seed=np.int64(seed), hc_count=np.int64(HC), hidden=np.int64(H), lowrank=np.int64(R),
        w_norm=_np(m.hc_norm.weight),                       # zero-centered (HF); gguf = 1+w
        w_down=_np(m.input_mix_weight_down.weight),         # [R, HC*H]
        w_up=_np(m.input_mix_weight_up.weight),             # [HC*H, R]
        w_inject=_np(m.block_inject_weight.weight),         # [HC, HC*H]
        x_wide=_np(x)[0], block_out=_np(block_out)[0],
        mixed=_np(mixed)[0], inject_w=_np(inj_w)[0], combined=_np(combined)[0],
    )
    return "hc.npz"


def dump_deltanet_step(mod, out_dir, seed):
    import torch

    gen = torch.Generator().manual_seed(seed)
    B, T, HV, DK, DV = 1, 8, 2, 8, 8
    q = torch.randn(B, T, HV, DK, generator=gen)
    k = torch.randn(B, T, HV, DK, generator=gen)
    v = torch.randn(B, T, HV, DV, generator=gen)
    g = -torch.rand(B, T, HV, generator=gen) * 2.0        # log-decay, negative
    beta = torch.rand(B, T, HV, generator=gen)
    s0 = torch.randn(B, HV, DK, DV, generator=gen) * 0.3  # nonzero initial state
    with torch.no_grad():
        out_rec, state_rec = mod.torch_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=s0, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        out_chunk, state_chunk = mod.torch_chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, chunk_size=4, initial_state=s0,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
    np.savez(
        os.path.join(out_dir, "deltanet_step.npz"),
        seed=np.int64(seed), heads=np.int64(HV), dk=np.int64(DK), dv=np.int64(DV),
        q=_np(q)[0], k=_np(k)[0], v=_np(v)[0], g=_np(g)[0], beta=_np(beta)[0],
        initial_state=_np(s0)[0],
        out_recurrent=_np(out_rec)[0], state_recurrent=_np(state_rec)[0],
        out_chunked=_np(out_chunk)[0], state_chunked=_np(state_chunk)[0],
    )
    return "deltanet_step.npz"


def dump_ple(mod, out_dir, seed):
    import torch

    gen = torch.Generator().manual_seed(seed)
    T = 12
    cfg = ref.Qwen4ExpTextConfig(
        hidden_size=16, hc_count=4, ple_embed_dim=32, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=2, vocab_size=1000, ngram_vocab_size_base=101,
        make_ngram_vocab_size_divisible_by=8, seed=1234, eos_token_id=7,
        rms_norm_eps=1e-6, ple_layer_ids=[1],
    )
    m = mod.Qwen4ExpTextPLELayer(cfg, layer_idx=0, ple_layer_index=0).float().eval()
    _fill_params(m, gen, torch)
    ids = torch.randint(0, cfg.vocab_size, (1, T), generator=gen)
    ids[0, T // 2] = cfg.eos_token_id  # exercise the n-gram EOS reset
    x_wide = torch.randn(1, T, cfg.hc_count * cfg.hidden_size, generator=gen)
    caps = {}
    h = m.ple_embedding.register_forward_hook(lambda _m, _i, o: caps.update(E=o))
    with torch.no_grad():
        out = m(x_wide, ids, past_key_values=None, conv_mask=None)
    h.remove()
    ng = m.ple_embedding
    np.savez(
        os.path.join(out_dir, "ple.npz"),
        seed=np.int64(seed), hidden=np.int64(cfg.hidden_size), hc_count=np.int64(4),
        ple_embed_dim=np.int64(cfg.ple_embed_dim), ngram_size=np.int64(3),
        heads_per_ngram=np.int64(2), vocab_size=np.int64(cfg.vocab_size),
        eos_token_id=np.int64(cfg.eos_token_id),
        conv_kernel=np.int64(cfg.ple_conv_kernel_size), conv_dilation=np.int64(cfg.ngram_size),
        layer_multipliers=ng.layer_multipliers.numpy().astype(np.uint64),
        head_vocab_sizes=ng.ngram_heads_vocab_sizes.numpy().astype(np.int64),
        head_offsets=ng.ngram_heads_offsets.numpy().astype(np.int64),
        embedding_table=_np(ng.ngram_embedding.weight),     # [padded_vocab, head_dim]
        w_key=_np(m.key_proj.weight), w_value=_np(m.value_proj.weight),
        w_norm_key=_np(m.norm_key.weight), w_norm_query=_np(m.norm_query.weight),
        w_norm_conv=_np(m.norm_conv.weight),                # all zero-centered (HF)
        w_conv=_np(m.conv1d.weight),                        # [C, 1, K]
        input_ids=ids[0].numpy().astype(np.int64), x_wide=_np(x_wide)[0],
        E=_np(caps["E"])[0], out=_np(out)[0],
    )
    return "ple.npz"


def dump_moe(mod, out_dir, seed):
    import torch

    gen = torch.Generator().manual_seed(seed)
    T = 6
    cfg = ref.Qwen4ExpTextConfig(
        hidden_size=32, num_experts=8, num_experts_per_tok=3, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, intermediate_size=16, hidden_act="silu",
        norm_topk_prob=True, _experts_implementation="eager",
    )
    m = mod.Qwen4ExpTextSparseMoeBlock(cfg).float().eval()
    _fill_params(m, gen, torch)
    x = torch.randn(1, T, cfg.hidden_size, generator=gen)
    caps = {}
    hooks = [
        m.gate.register_forward_hook(lambda _m, _i, o: caps.update(
            logits=o[0], top_w=o[1], top_idx=o[2])),
        m.shared_expert.register_forward_hook(lambda _m, _i, o: caps.update(shared=o)),
        m.experts.register_forward_hook(lambda _m, _i, o: caps.update(routed=o)),
    ]
    with torch.no_grad():
        out = m(x)
    for h in hooks:
        h.remove()
    np.savez(
        os.path.join(out_dir, "moe.npz"),
        seed=np.int64(seed), hidden=np.int64(cfg.hidden_size),
        num_experts=np.int64(cfg.num_experts), top_k=np.int64(cfg.num_experts_per_tok),
        moe_ffn=np.int64(cfg.moe_intermediate_size),
        norm_topk_prob=np.int64(1),
        w_router=_np(m.gate.weight),                        # [E, H]
        w_gate_up=_np(m.experts.gate_up_proj),              # [E, 2*F, H]
        w_down=_np(m.experts.down_proj),                    # [E, H, F]
        w_sh_gate=_np(m.shared_expert.gate_proj.weight),
        w_sh_up=_np(m.shared_expert.up_proj.weight),
        w_sh_down=_np(m.shared_expert.down_proj.weight),
        w_shexp_gate=_np(m.shared_expert_gate.weight),      # [1, H]
        x=_np(x)[0],
        router_logits=_np(caps["logits"]), router_top_w=_np(caps["top_w"]),
        router_top_idx=caps["top_idx"].numpy().astype(np.int64),
        routed_out=_np(caps["routed"]), shared_out=_np(caps["shared"]),
        out=_np(out)[0],
    )
    return "moe.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    os.makedirs(args.out_dir, exist_ok=True)
    mod = ref.load_modeling()

    written = []
    for fn in (dump_hc, dump_deltanet_step, dump_ple, dump_moe):
        written.append(fn(mod, args.out_dir, args.seed))

    bad = []
    for name in written:
        z = np.load(os.path.join(args.out_dir, name))
        for k in z.files:
            a = z[k]
            if a.dtype.kind == "f":
                if not np.isfinite(a).all():
                    bad.append(f"{name}:{k} non-finite")
                elif a.size > 4 and float(a.std()) == 0.0:
                    bad.append(f"{name}:{k} zero variance")
        print(f"{name}: keys={z.files}")
    print("STATUS:", "OK" if not bad else "; ".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
