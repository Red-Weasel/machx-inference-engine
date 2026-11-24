import sys, numpy as np
D = os.path.join(os.environ.get("IE_DATA_DRIVE", os.path.expanduser("~/data")), "models/Xing4.0-29B-A4B-GGUF-ie/gptq-work/heldout/")
import os
X = os.environ.get("XG4_WORK", "/tmp/xing4-heldout") + "/"
V = 131072; R = 1586
mask = np.fromfile(X + "heldout.mask", np.uint8); T = len(mask)
call = mask[T - R + 1:T].astype(bool)          # row i predicts token i+1: rows whose next token sits in a tool call
call = np.append(call, False)
def lsm(x):
    m = x.max(-1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))
ref = lsm(np.fromfile(D + "q8_0.f32", np.float32).reshape(R, V).astype(np.float64))
for q in sys.argv[1:]:
    e = lsm(np.fromfile(D + q + ".f32", np.float32).reshape(R, V).astype(np.float64))
    kl = (np.exp(ref) * (ref - e)).sum(-1)
    same = ref.argmax(-1) == e.argmax(-1)
    print(f"{q:18s} all rows KL {kl.mean():.5f} (top token same {same.mean()*100:.1f} %)   tool-call rows KL {kl[call].mean():.5f} (same {same[call].mean()*100:.1f} %, {call.sum()} rows)")
