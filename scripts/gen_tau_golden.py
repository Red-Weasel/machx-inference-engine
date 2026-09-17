# Golden vectors for ink_log_scaling_tau, from modeling_inkling.py:255-258 verbatim.
import struct, sys, torch
out = open(sys.argv[1], "wb")
cases = [(8, 0.1, 128000.0, 0), (16, 0.1, 128000.0, 200000),
         (32, 0.1, 128000.0, 127990), (4, 0.25, 100.0, 0), (5, 0.1, 128000.0, 1000000)]
out.write(struct.pack("<I", len(cases)))
for (T, alpha, n_floor, off) in cases:
    q_positions = torch.arange(T, dtype=torch.long) + off
    # ---- reference ----
    effective_n = (q_positions + 1).float()
    tau = 1.0 + alpha * torch.log((effective_n / n_floor).clamp(min=1.0))
    # ---- /reference ----
    out.write(struct.pack("<Iff", T, alpha, n_floor))
    out.write(q_positions.numpy().astype("<i4").tobytes())
    out.write(tau.numpy().astype("<f4").tobytes())
out.close()
print("golden written:", sys.argv[1])
