# Soak gate

`soak.sh` starts `ie serve` with a real launch line, runs `soak_serve.py` against it for N minutes, stops the server the
orderly way (SIGTERM) and reads its log. It exits 0 only if every check held.

```
scripts/soak/soak.sh <gguf> <minutes> [out_dir] -- <serve args...>
```

The traffic (`soak_serve.py`, usable alone against any running server):

| Stream | What it does | What must hold |
|---|---|---|
| Conversation A | One conversation that grows turn by turn (greedy). Every 5th turn an EARLIER message is trimmed in place, every 7th the last assistant message is edited; above 150K prompt tokens it starts over. | Every reply arrives. The server log shows `restart point at ...` for the rewrites instead of full re-reads. |
| Job B | The same ~15K-token greedy prompt, again and again, on another lane. | Every reply byte-identical to the first. |
| Chaos | Every 40-90 s: a streamed request the client abandons, or three short requests at once. | The server keeps answering. |

It fails on: a missing or empty reply, job B's reply changing, any GPU fault / reset line in the kernel log
(`journalctl -k`), the server's RSS growing more than 6 GiB, the server gone, or a `DEVICE_LOST` / `generation error` /
`EXCEPTION` line in the server log. `soak.json` in the output directory holds the counts and the per-turn seconds.

Run it before a release, on the launch line people actually use.
