# Golden vectors for ink_shortconv, produced by the REFERENCE code path itself
# (torch F.conv1d with the module's exact padding/truncation), not by a
# re-derivation. Writes a flat binary: T,C,K then x[T*C], w[K*C], y[T*C].
import struct, sys
import torch, torch.nn.functional as F
torch.manual_seed(11)
out = open(sys.argv[1], "wb")
cases = [(1,8,4),(4,8,4),(7,16,4),(16,4096,4),(1024,64,4),(3,5,4)]
out.write(struct.pack("<I", len(cases)))
for (T, C, K) in cases:
    x = torch.randn(1, T, C, dtype=torch.float32)
    # [C, K] — the GGUF layout (channel-major, taps contiguous). The engine
    # kernel consumes this verbatim; torch's conv1d wants [C, 1, K].
    w = torch.randn(C, K, dtype=torch.float32) * 0.3
    # EXACT reference sequence: modeling_inkling.py:510-542
    hs = x.float()
    residual = hs
    hs = hs.transpose(1, 2)                       # [1, C, T]
    padding = K - 1
    conv = F.conv1d(hs, weight=w.unsqueeze(1), bias=None,
                    padding=padding, groups=C)[:, :, :T]
    y = (conv.transpose(1, 2) + residual)
    out.write(struct.pack("<III", T, C, K))
    out.write(x.contiguous().numpy().astype("<f4").tobytes())
    out.write(w.contiguous().numpy().astype("<f4").tobytes())
    out.write(y.contiguous().numpy().astype("<f4").tobytes())
out.close()
print("golden written:", sys.argv[1])
