#!/usr/bin/env python3
"""Measure what the 9100 PRO actually delivers at DeepSeek-V4.1 expert granularity.

An expert is three FP4 planes of 5.625 MiB each (2304x2560 and 5120x1152, both 5,898,240 B)
plus small scale planes. Decode touches 6 experts x 40 layers = 240 expert reads per token,
scattered across 48 shards. So the number that matters is RANDOM ~5.6 MiB reads, cold.

O_DIRECT bypasses the page cache, so this measures the drive, not RAM. Read-only; opens the
shipped shards and never writes.
"""
import os, sys, time, random, ctypes, glob
from concurrent.futures import ThreadPoolExecutor

D = os.path.expanduser("~/models/DeepSeek-V4.1-Flash")
PLANE = 5_898_240                      # one FP4 expert plane, bytes
ALIGN = 4096
files = sorted(glob.glob(f"{D}/model-*.safetensors"))
sizes = [os.path.getsize(f) for f in files]
total = sum(sizes)
print(f"{len(files)} shards, {total/2**30:.1f} GiB, plane {PLANE/2**20:.3f} MiB")

libc = ctypes.CDLL("libc.so.6", use_errno=True)
def aligned_buf(n):
    p = ctypes.c_void_p()
    if libc.posix_memalign(ctypes.byref(p), ctypes.c_size_t(ALIGN), ctypes.c_size_t(n)) != 0:
        raise MemoryError
    return (ctypes.c_char * n).from_address(p.value)

def run(threads, n_reads, rounds=3):
    fds = [os.open(f, os.O_RDONLY | os.O_DIRECT) for f in files]
    try:
        best = 0.0
        for _ in range(rounds):
            rnd = random.Random(1234)
            # round the read up to a 4K multiple; O_DIRECT needs aligned length+offset
            rlen = (PLANE + ALIGN - 1) // ALIGN * ALIGN
            jobs = []
            for _ in range(n_reads):
                i = rnd.randrange(len(files))
                off = rnd.randrange(0, max(1, sizes[i] - rlen)) // ALIGN * ALIGN
                jobs.append((i, off))
            def one(job):
                i, off = job
                buf = aligned_buf(rlen)
                got, pos = 0, off
                while got < rlen:
                    n = libc.pread(fds[i], ctypes.byref(buf, got), ctypes.c_size_t(rlen-got),
                                   ctypes.c_long(pos))
                    if n <= 0: break
                    got += n; pos += n
                return got
            t0 = time.perf_counter()
            if threads == 1:
                got = sum(one(j) for j in jobs)
            else:
                with ThreadPoolExecutor(threads) as ex:
                    got = sum(ex.map(one, jobs))
            dt = time.perf_counter() - t0
            best = max(best, got / dt)
        return best
    finally:
        for fd in fds: os.close(fd)

print(f"\n{'threads':>8s} {'GB/s':>8s} {'planes/s':>10s} {'ms/expert(3 planes)':>20s}")
for th in (1, 2, 4, 8, 16, 32):
    bw = run(th, max(64, th*16))
    pps = bw / PLANE
    print(f"{th:8d} {bw/1e9:8.2f} {pps:10.0f} {3000/pps:20.2f}")
