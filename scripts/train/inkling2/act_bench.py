"""The experts' softcapped SwiGLU: eager vs torch.compile(dynamic), forward + backward, bit-exactness and time.

  xpu_env.sh venv/bin/python -I act_bench.py
"""
import time
import torch
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopSoftcappedSwiGLU

dev = "xpu:0"
cfg = Inkling2LoopTextConfig(mlp_gate_softcap=4.0, mlp_up_softcap=25.0)
act = Inkling2LoopSoftcappedSwiGLU(cfg).to(dev)
comp = torch.compile(act, dynamic=True)
for n in (300, 700, 1500):
    g = (torch.randn(32, n, 1152, device=dev) * 3).bfloat16().requires_grad_()
    u = (torch.randn(32, n, 1152, device=dev) * 3).bfloat16().requires_grad_()
    w = torch.randn(32, n, 1152, device=dev).bfloat16()
    res = {}
    for name, f in (("eager", act), ("compiled", comp)):
        y = f(g, u)
        dg, du = torch.autograd.grad((y.float() * w.float()).sum(), (g, u))
        res[name] = (y, dg, du)
        torch.xpu.synchronize()
        t = time.time()
        for _ in range(10):
            y = f(g, u)
            torch.autograd.grad((y * w).sum(), (g, u))
        torch.xpu.synchronize()
        res[name + "_ms"] = (time.time() - t) * 100
    same = [torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(res["eager"], res["compiled"])]
    mx = [(a.float() - b.float()).abs().max().item() for a, b in zip(res["eager"], res["compiled"])]
    print(f"n={n}: eager {res['eager_ms']:.1f} ms, compiled {res['compiled_ms']:.1f} ms (fwd+bwd); bit-identical "
          f"y/dgate/dup {same}; max abs diff {mx}", flush=True)
