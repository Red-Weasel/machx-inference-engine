#!/usr/bin/env python3
"""Launches per layer of one decode step (V4.1 Phase 11, docs/deepseek41/29 criterion 5).

Run the decode test with the runtime's UR trace and the test's layer markers, stderr to a file:
    IE_DS41_LAUNCH_MARKS=1 SYCL_UR_TRACE=2 ie-ds41-decode-test ... 2> trace.txt
then  ur_launch_count.py trace.txt
Counts, between "@@step begin" and "@@step end", every UR enqueue / host wait / USM alloc call
per bucket; bucket L holds the calls issued between the marker of layer L-1 and that of layer L
(bucket 0 and bucket 20 therefore include their card's setup: rope tables, scratch, stream
copies; the bucket after the last layer is the head)."""
import collections, re, sys

# SYCL_UR_TRACE=2 with the 2026.1 Level Zero adapter prints the adapter's INTERNAL calls ("UR --->
# hKernel->prepareForSubmission(...)"), not the loader's urEnqueue* names: one prepareForSubmission
# per kernel launch; queueGetInfo once per enqueue of any kind (kernel, memcpy, fill), so its excess
# over the kernels counts the copies and fills; releaseSubmittedKernels once per host drain.
pat = re.compile(r'^UR ---> (?:hKernel->(prepareForSubmission)|hQueue->(queueGetInfo)|lockedCommandListManager->(releaseSubmittedKernels))\(')
buckets = collections.OrderedDict(); cur = None; inside = False; seen_arrow = False
for line in open(sys.argv[1], errors='replace'):
    if line.startswith('@@step begin'):
        inside = True; cur = 'L0'; buckets[cur] = collections.Counter(); continue
    if line.startswith('@@step end'):
        inside = False; continue
    if not inside: continue
    m = re.match(r'@@layer (\d+)', line)
    if m:
        cur = 'L%d' % (int(m.group(1)) + 1); buckets[cur] = collections.Counter(); continue
    if line.startswith('UR --->'): seen_arrow = True
    m2 = pat.match(line)
    if m2: buckets[cur][next(g for g in m2.groups() if g)] += 1
if not buckets: sys.exit('no "@@step begin" marker in %s' % sys.argv[1])
n_layers = len(buckets) - 1
last = list(buckets)[-1]; buckets['head'] = buckets.pop(last)
def cls(c):
    k = c.get('prepareForSubmission', 0)
    cp = max(0, c.get('queueGetInfo', 0) - k)      # copies + fills (queueGetInfo per enqueue, minus the kernels)
    fl = 0; ot = 0
    w = c.get('releaseSubmittedKernels', 0)         # host drains
    al = 0
    return k, cp, fl, ot, w, al
print('%-6s %7s %6s %5s %5s %5s %6s %8s' % ('bucket', 'kernels', 'cp+fill', '-', '-', 'drains', '-', 'enqueues'))
tot = collections.Counter(); per_layer = []
for b, c in buckets.items():
    k, cp, fl, ot, w, al = cls(c)
    print('%-6s %7d %6d %5d %5d %5d %6d %8d' % (b, k, cp, fl, ot, w, al, k + cp + fl + ot))
    if b != 'head': per_layer.append((k + cp + fl + ot, k, w))
    tot.update(c)
k, cp, fl, ot, w, al = cls(tot)
enq = sorted(x[0] for x in per_layer); ker = sorted(x[1] for x in per_layer); wa = sorted(x[2] for x in per_layer)
print('step: %d enqueues (%d kernels, %d memcpy, %d fill, %d other), %d host waits, %d USM alloc/free calls, over %d layer buckets + head'
      % (k + cp + fl + ot, k, cp, fl, ot, w, al, n_layers))
print('per layer bucket, median: %d enqueues (%d kernels), %d host waits; min/max enqueues %d/%d'
      % (enq[len(enq) // 2], ker[len(ker) // 2], wa[len(wa) // 2], enq[0], enq[-1]))
if not seen_arrow: print('note: no "UR --->" trace lines seen; check that SYCL_UR_TRACE=2 was set and stderr captured')
