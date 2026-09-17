"""Load the HF reference modeling_qwen4_exp.py into a transformers 5.12.0 venv.

The reference file (~/references/qwen38-flash-next/modeling_qwen4_exp.py)
was generated for a newer transformers than the parity venv ships (5.12.0), and its
companion configuration_qwen4_exp.py is not on disk.  This module:

  1. registers a synthetic package  transformers.models.qwen4_exp  so the file's
     relative imports (`from ... import X`) resolve against the installed transformers;
  2. shims the handful of symbols 5.12.0 lacks (all are either identity decorators
     whose fallback IS the reference torch math, or functions used only on paths a
     single-layer oracle never executes -- full-model forward and the vision tower);
  3. provides Qwen4ExpTextConfig as a plain attribute container built from
     hf-modeling/config.json's text_config.

EVERY shim is listed in SHIMS_APPLIED after load_modeling() runs, and documented in
README.md.  No shim replaces any math that a decoder-layer forward executes.

Run under ~/venv/bin/python (torch 2.12.0+cpu, transformers 5.12.0).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import types

MODELING_PATH = os.path.expanduser("~/references/qwen38-flash-next/modeling_qwen4_exp.py")
CONFIG_JSON = os.path.expanduser("~/references/qwen38-flash-next/hf-modeling/config.json")

# Extra sys.path entries that provide the pure-python `gguf` package (the parity venv
# lacks it; vllm-xpu's site-packages carries gguf 2.2.6).  Appended (not prepended),
# so the venv's own numpy/torch always win.
GGUF_FALLBACK_PATHS = [
    os.path.expanduser("~/vllm-xpu/lib/python3.12/site-packages"),
    os.path.expanduser("~/vllm-env/lib/python3.12/site-packages"),
]

SHIMS_APPLIED: list[str] = []

_MODULE = None


def import_gguf():
    try:
        import gguf  # noqa: F401
    except ImportError:
        for p in GGUF_FALLBACK_PATHS:
            if p not in sys.path:
                sys.path.append(p)
        import gguf  # noqa: F401
        SHIMS_APPLIED.append(
            f"sys.path += {GGUF_FALLBACK_PATHS[0]} (gguf package only; appended last)"
        )
    import gguf
    return gguf


def _identity_decorator_factory(*_a, **_k):
    def deco(fn):
        return fn
    return deco


def _install_shims():
    import torch
    import transformers
    import transformers.utils.logging as _tlog

    # silence auto_docstring [ERROR] spam printed while executing the modeling file
    # (docstring lint of the vision/CausalLM classes; cosmetic only)
    _tlog.set_verbosity(50)

    # transformers.initialization exists in 5.12.0 (lazy submodule); make sure the
    # four helpers _init_weights would use exist.  We never call _init_weights, but
    # the module-level `from ... import initialization as init` must succeed.
    try:
        ti_init = importlib.import_module("transformers.initialization")
        import torch.nn.init as _tni

        def _copy_(dst, src):
            with torch.no_grad():
                dst.copy_(src)
            return dst

        for fn, fallback in (
            ("ones_", _tni.ones_),
            ("zeros_", _tni.zeros_),
            ("normal_", _tni.normal_),
            ("copy_", _copy_),
        ):
            if not hasattr(ti_init, fn):
                setattr(ti_init, fn, fallback)
                SHIMS_APPLIED.append(f"transformers.initialization.{fn} -> torch.nn.init fallback")
    except ImportError:
        m = types.ModuleType("transformers.initialization")
        import torch.nn.init as _tni

        def _copy_(dst, src):
            with torch.no_grad():
                dst.copy_(src)
            return dst

        m.ones_, m.zeros_, m.normal_, m.copy_ = _tni.ones_, _tni.zeros_, _tni.normal_, _copy_
        sys.modules["transformers.initialization"] = m
        transformers.initialization = m
        SHIMS_APPLIED.append("transformers.initialization -> module shim (torch.nn.init)")

    import transformers.integrations as ti
    if not hasattr(ti, "use_kernel_func_from_hub_with_fallback"):
        # decorates causal_conv1d_* and the two gated-delta-rule functions; with no
        # kernels hub the real decorator would run the decorated torch fallback --
        # identity keeps exactly the reference math in the file.
        ti.use_kernel_func_from_hub_with_fallback = _identity_decorator_factory
        SHIMS_APPLIED.append(
            "integrations.use_kernel_func_from_hub_with_fallback -> identity "
            "(reference torch fallback runs)"
        )

    import transformers.integrations.accelerate as ta
    if not hasattr(ta, "force_accelerate_hooks"):
        # decorates GatedDeltaNet.forward; only matters under accelerate device_map
        ta.force_accelerate_hooks = _identity_decorator_factory
        SHIMS_APPLIED.append("integrations.accelerate.force_accelerate_hooks -> identity")

    import transformers.masking_utils as tmask
    if not hasattr(tmask, "create_recurrent_attention_mask"):
        # used only inside Qwen4ExpTextModel.forward / full-model paths (never called
        # by the single-layer oracle, which builds its own masks)
        def create_recurrent_attention_mask(**_kw):
            raise RuntimeError("shim: create_recurrent_attention_mask is full-model-only")

        tmask.create_recurrent_attention_mask = create_recurrent_attention_mask
        SHIMS_APPLIED.append("masking_utils.create_recurrent_attention_mask -> stub (never called)")

    import transformers.utils.generic as tgen
    if not hasattr(tgen, "get_max_seqlen"):
        # used only on the vision flash-attention path
        def get_max_seqlen(*_a, **_k):
            raise RuntimeError("shim: get_max_seqlen is a vision/flash-only helper")

        tgen.get_max_seqlen = get_max_seqlen
        SHIMS_APPLIED.append("utils.generic.get_max_seqlen -> stub (vision/flash path only)")

    import transformers.vision_utils as tvis
    for fn in ("get_vision_attention_seqlens", "get_vision_interpolation_indices_and_weights"):
        if not hasattr(tvis, fn):
            def _stub(*_a, **_k):
                raise RuntimeError("shim: vision tower is not supported by the oracle")

            setattr(tvis, fn, _stub)
            SHIMS_APPLIED.append(f"vision_utils.{fn} -> stub (vision tower never instantiated)")


class Qwen4ExpTextConfig:
    """Plain attribute container standing in for the missing configuration file.

    The decoder-layer / PLE / rotary classes only read attributes off the config, so
    a namespace is sufficient; none of the PreTrainedConfig machinery is exercised.
    """

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class Qwen4ExpConfig(Qwen4ExpTextConfig):
    pass


class Qwen4ExpVisionConfig(Qwen4ExpTextConfig):
    pass


def load_modeling():
    """Exec the reference modeling file as transformers.models.qwen4_exp.modeling_qwen4_exp."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    _install_shims()
    import transformers

    cfgmod = types.ModuleType("transformers.models.qwen4_exp.configuration_qwen4_exp")
    cfgmod.Qwen4ExpConfig = Qwen4ExpConfig
    cfgmod.Qwen4ExpTextConfig = Qwen4ExpTextConfig
    cfgmod.Qwen4ExpVisionConfig = Qwen4ExpVisionConfig
    SHIMS_APPLIED.append(
        "configuration_qwen4_exp -> plain-namespace config classes (file not on disk)"
    )

    pkg = types.ModuleType("transformers.models.qwen4_exp")
    pkg.__path__ = []
    sys.modules["transformers.models.qwen4_exp"] = pkg
    sys.modules["transformers.models.qwen4_exp.configuration_qwen4_exp"] = cfgmod
    setattr(transformers.models, "qwen4_exp", pkg)
    pkg.configuration_qwen4_exp = cfgmod

    spec = importlib.util.spec_from_file_location(
        "transformers.models.qwen4_exp.modeling_qwen4_exp", MODELING_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MODULE = mod
    return mod


def build_text_config(**overrides):
    """Qwen4ExpTextConfig from hf-modeling/config.json text_config.

    Adds the attributes the modeling code reads but config.json omits:
      seed=1234            VERIFIED: _build_layer_multipliers(248320, 3, 0, 1234)
                           reproduces the GGUF's qwen4exp.ple.layer_multipliers
                           [23703573157769, 20109073645365, 8052911324071] exactly
                           (brute-forced over seeds 0..200000).
      norm_topk_prob=True  ASSUMPTION (documented): config.json lacks the key and the
                           HF configuration file is not on disk; llama.cpp PR27742
                           hardcodes renormalize=true, so golden tensors renormalize
                           the top-10 weights to sum 1.
      intermediate_size    = shared_expert_intermediate_size (only read as a default
                           the model never uses: every MLP instantiation passes an
                           explicit intermediate_size).
      _attn_implementation='eager', _experts_implementation='eager'
    """
    raw = json.load(open(CONFIG_JSON))["text_config"]
    raw.pop("mtp", None)
    cfg = Qwen4ExpTextConfig(**raw)
    cfg.seed = 1234
    cfg.norm_topk_prob = True
    cfg.intermediate_size = raw["shared_expert_intermediate_size"]
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_position_embeddings(mod, cfg, T, device="cpu"):
    """(cos, sin), each [1, T, 64] fp32, exactly as Qwen4ExpTextModel would produce.

    The model expands 2D position_ids [1, T] to three identical T/H/W streams
    (text-only), which the rotary forward does itself for ndim==2 input.
    """
    import torch

    rot = mod.Qwen4ExpTextRotaryEmbedding(cfg)
    pos = torch.arange(T, device=device).unsqueeze(0)
    anchor = torch.empty(1, T, cfg.hidden_size, dtype=torch.float32, device=device)
    return rot(anchor, pos)


def make_full_attn_mask(T, device="cpu"):
    """Eager 4D float causal mask [1, 1, T, T]: 0 where visible, finfo.min above diag.

    Matches what create_causal_mask produces for the eager implementation with no
    padding; the QSA indexer recovers visibility via (mask == 0).
    """
    import torch

    m = torch.full((1, 1, T, T), torch.finfo(torch.float32).min, device=device)
    return torch.triu(m, diagonal=1)
