"""Expert decode: eager vs torch.compile, bit-exactness and time, on one layer's experts (no model load).

  xpu_env.sh venv/bin/python -I decode_bench.py MODEL_DIR xpu:0
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from inkling2.compressed_experts import CompressedExperts, SafetensorsReader

model_dir, dev = sys.argv[1], sys.argv[2]
r = SafetensorsReader(os.path.join(model_dir, "model-00001-of-00001.safetensors"))
for layer in (0, 20):
    ce = CompressedExperts(r, layer, None, amax="exact").to(dev)
    idx = torch.arange(128, device=dev)
    comp = torch.compile(ce.decode.__func__, dynamic=False)
    for short in ("w13", "w2"):
        a = ce.decode(short, idx)
        b = comp(ce, short, idx)
        same = torch.equal(a.view(torch.int16), b.view(torch.int16))
        for f, name in ((lambda: ce.decode(short, idx), "eager"), (lambda: comp(ce, short, idx), "compiled")):
            f(); torch.xpu.synchronize(dev)
            t = time.time()
            for _ in range(5):
                f()
            torch.xpu.synchronize(dev)
            print(f"layer {layer} {'NVFP4' if ce.nvfp4 else 'MXFP8'} {short}: {name} {1000 * (time.time() - t) / 5:.1f} ms", flush=True)
        print(f"layer {layer} {short}: compiled bit-identical to eager: {same}", flush=True)
    del ce
print(f"peak {torch.xpu.max_memory_allocated(dev) / 2**30:.2f} GiB")
