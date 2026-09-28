# #29: the idle-spin watchdog in `ie serve`

2026-09-24, branch `engine-fix-48-29-54`. **Status: code + CPU tests; the GPU steps are in the runbook**
(`docs/RUNBOOK_2026-09-24_fixes-48-29-54.md`, section D).

## The incident, and why there is no fix for the cause

On 2026-09-19 at 23:00 a client disconnected mid-generation. The server printed
`client disconnected mid-generation — aborted after 1589 tok`, and then burned ~900 % CPU for eight hours with
inflight=0 and queued=0. It grew from ~6 to 18.7 GiB RSS, filled the 8 GiB of swap, and held the port.

One deliberate repro on 09-21 did not spin. Two changes landed afterwards and may or may not have removed the cause:

* `kmp_set_blocktime(0)` on request threads (4cfd9cd);
* the fill-event leak fix (4cd3cd8).

The fix list says it plainly: not safe to guess-patch without a stack.

So this change does not claim a fix. It makes the next occurrence **self-diagnosing** (one log line that names the busy
threads and says how to take their stacks) and **bounded** (the line repeats at most every 10 minutes, and `/health`
exposes the state so a supervisor can restart the process).

## What it does

`run_openai_server`'s watcher thread already woke every 100 ms. It now also takes a sample every 5 s:

* the process's utime + stime from `/proc/self/stat`;
* whether the server was **idle over the whole interval**: `inflight == 0`, `queued == 0`, and `Admission::admitted()`
  unchanged. `admitted()` is a new monotonic count of requests that got a run slot. Without it, a request that started
  and finished between two samples would look idle, and its CPU would count as a spin.

`IdleSpinDetector` (`include/ie/idle_spin.hpp`) keeps the idle samples. Once the server has been idle for a whole
**60 s window**, it computes the CPU rate over that window in cores. When the rate is **at least the threshold**
(`IE_IDLE_SPIN_CORES`, default 2.0) it reports once, then again every **600 s** while the rate stays above the threshold.
Any activity, or the rate falling below the threshold, ends the spin state, so a later spin is reported at once.

The report is one stderr line:

    [ie] idle spin: 9.00 cores for 65 s with nothing in flight, queued or admitted (threshold 2.00 cores over 60 s;
    report 1, again every 600 s while it lasts; IE_IDLE_SPIN_WATCHDOG=0 turns this off) | 342 threads (ie x340, ...),
    OpenMP default team 20 | busiest over the last 5.0 s: tid 12345 "ie" R 0.99 cores (usr 0.40 sys 0.59) wchan 0
    [http request thread]; tid ... | stacks: eu-stack -p 4242 (or gdb -p 4242 -batch -ex 'thread apply all bt')

(It is shown wrapped here; the real one is a single line.) Per thread it gives:

* **usr / sys split.** A thread spinning in user space (an OpenMP barrier, a busy-wait on an event) shows usr; a thread
  looping on syscalls that return at once (a poll on a dead socket) shows sys. That alone separates the two leading
  guesses.
* **wchan**: where the thread sleeps in the kernel; `0` while it runs.
* **`[http request thread]`**: set when the thread is one of the HTTP pool's threads that ran a chat request.
  `note_http_thread()` records them. Every thread inherits its creator's name, so most read "ie" and the name alone
  cannot tell them apart.
* **the thread count and the three most common names**, and OpenMP's default team size.

Unverified: whether libiomp5 or the Level Zero runtime rename their threads is unknown to me, so the name histogram may
say only "ie x N". oneDNN exposes no thread-count API for its GPU path, so the "OpenMP/oneDNN thread counts" of the task
are reported as the thread count, the names and OpenMP's default team.

`/health` gains:

    "idle_spin": {"seconds": N, "cores": X}

* `seconds`: how long the current spin has lasted, measured from the start of the first 60 s window that reached the
  threshold. 0 when there is no spin.
* `cores`: the rate over the latest full idle window. 0 while the server has not been idle for a whole window.

The field is absent when the watchdog is off. The status stays "ok": the field is information for a supervisor, and a
Dream-side restart policy on `seconds` is the obvious next step (not done here).

## Bounds

* The sample is two `/proc` reads every 5 s.
* The per-thread snapshot (`/proc/self/task/*/stat`) is taken on idle samples only and reads at most 4,096 threads; the
  rest are counted. The 09-16 leak reached 94,178 threads.
* The report runs only when due: the first detection, then every 600 s.
* Nothing is killed or restarted, and no engine state is touched.

## Knobs

| variable | effect |
|---|---|
| `IE_IDLE_SPIN_WATCHDOG=0` | off: no sampling, no line, no `/health` field |
| `IE_IDLE_SPIN_CORES=<x>` | the threshold in cores, `>= 0`. A value that is not a number >= 0 keeps 2.0 and prints a warning. **`0` reports every idle minute: that is the plumbing check of the runbook** |

The startup line says which applies: `[ie] idle-spin watchdog on: ...` or `... off (IE_IDLE_SPIN_WATCHDOG=0)`.

Why 2.0 cores by default: the V4.1 server idles at ~0 % (the 09-21 repro: "0 % CPU, inflight 0, 150 threads"), the
incident was ~9 cores, and a threshold above one core ignores a single legitimately busy background thread. Examples of
such a thread are the prefix cache's disk writer and `malloc_trim` after a reply, both of which last well under a
second. To catch one spinning thread, set 0.8. Unverified: the other engines' idle baselines (DS4, GLM, Qwen, MiMo) were
not measured; a false report there costs one line per 10 minutes.

## What the next occurrence gives, and what to do with it

1. The line above, within ~65 s of the spin starting.
2. At once, the stacks: `eu-stack -p <pid>` (or the gdb form) on the TIDs the line names. The fix list's standing
   request is "a repro with a stack".
3. Then `POST /admin/shutdown`.

## Files

* `include/ie/idle_spin.hpp`, `src/server/idle_spin.cpp`: the detector, the `/proc` readers, the report line. No SYCL.
* `include/ie/server_admission.hpp`: `Admission::admitted()`.
* `src/server/openai_server.cpp`: the sampler in the watcher thread, the `/health` field, the HTTP-thread registry, the
  startup line. The functions touched are `run_openai_server` and the new `note_http_thread`.
* `tests/unit/idle_spin_test.cpp`: 53 checks. They cover configuration parsing; the detector on fake sample streams (a
  quiet idle process, the 09-19 shape, a request mid-window, a request mid-spin, the inclusive threshold, a spin that
  stops and restarts, irregular sample times, disabled, threshold 0, a counter that goes backwards); the stat-line
  parser (a comm holding spaces and ')'); the deltas; the real `/proc` of the test process with one thread spinning
  400 ms, which the readers must name as the busiest; and the report line's content.
* `tests/unit/server_admission_test.cpp`: `admitted()` counts runs, not refusals.
* `tools/serve_disconnect_probe.py`: the runbook's disconnect loop. It was checked on the CPU against a fake SSE server:
  a quiet server gives PASS; a server that starts a busy thread after two disconnects gives FAIL at 1.0 cores; a reply
  that finishes before the cut gives FAIL.
