#!/usr/bin/python3 -I
"""Run a command under a Landlock domain: ll_exec.py --rw DIR ... --ro PATH ... --port N -- argv...

Why Landlock and not a container: Dream's tools refuse to run unless the system Python is really root-owned, which a user
namespace hides. Landlock keeps every file's real owner and only takes access away. What the domain allows: read/execute
on the --ro paths, everything on the --rw paths, listing directory names anywhere (Dream walks the workspace's parents),
TCP connect to --port only, no TCP bind, no signals or abstract sockets outside the domain. Outbound UDP is not covered.
"""
import ctypes, os, sys

EXEC, WR, RD, RDIR = 1, 2, 4, 8
ALL_FS = (1 << 16) - 1
FILE_ONLY = EXEC | WR | RD | (1 << 14) | (1 << 15)
RW = ALL_FS & ~(64 | 2048)                   # no device nodes
RO = EXEC | RD | RDIR
libc = ctypes.CDLL(None, use_errno=True)


def die(what):
    sys.exit(f"ll_exec: {what}: {os.strerror(ctypes.get_errno())}")


def add_path(rs, path, rights):
    fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    if not os.path.isdir(path):
        rights &= FILE_ONLY
    rule = ctypes.create_string_buffer(int(rights).to_bytes(8, "little") + fd.to_bytes(4, "little", signed=True), 12)
    if libc.syscall(445, rs, 1, rule, 0):
        die(f"rule for {path}")
    os.close(fd)


args = sys.argv[1:]
cut = args.index("--")
opts, argv = args[:cut], args[cut + 1:]
rw, ro, ports, i = [], [], [], 0
while i < len(opts):
    {"--rw": rw, "--ro": ro, "--port": ports}[opts[i]].append(opts[i + 1])
    i += 2
attr = ctypes.create_string_buffer(ALL_FS.to_bytes(8, "little") + (3).to_bytes(8, "little") + (3).to_bytes(8, "little"), 24)
rs = libc.syscall(444, attr, 24, 0)
if rs < 0:
    die("create ruleset")
add_path(rs, "/", RDIR)
for p in ro:
    add_path(rs, p, RO)
for p in rw:
    add_path(rs, p, RW)
for p in ports:
    rule = ctypes.create_string_buffer((2).to_bytes(8, "little") + int(p).to_bytes(8, "little"), 16)
    if libc.syscall(445, rs, 2, rule, 0):
        die("port rule")
if libc.prctl(38, 1, 0, 0, 0):
    die("no_new_privs")
if libc.syscall(446, rs, 0):
    die("restrict")
os.execvp(argv[0], argv)
