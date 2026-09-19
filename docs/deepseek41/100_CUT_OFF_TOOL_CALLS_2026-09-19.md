# V4.1 — naming a tool call that the output limit cut off

## Why

Three times in the 2026-09-18/19 Dream session a reply ran to the 16,384-token output cap in the middle of a big
`write_file`; nothing was written and ~95 minutes of decode were lost. The server strips an unfinished DSML block
from the visible content (correctly: a truncated call must never run), so the client could not tell "the model
stopped" from "the model was cut off mid-call", and Dream's own "a tool call was cut off, write it in pieces"
notice never fired.

## What changed

- `ds41_cut_tool_name(text)` (src/model/deepseek41_prompt.cpp): for a reply that holds a DSML block, the name of the
  LAST invoke in it (the one the cut landed in), `"unknown"` before a name was written, `""` when there is no block.
  The call is still never parsed or run.
- `GenerateResult::truncated_tool_call`: set by the V4.1 chat path when `finish_reason == "length"`.
- The response carries it on the choice, in both paths:
  `{"finish_reason": "length", "truncated_tool_call": {"name": "write_file"}, ...}` (non-stream), and on the closing
  chunk of a stream. Standard clients ignore the extra field; Dream reads it (DREAM fix #8/#14).

## Gates

- `tools/ds41_reference/cut_tool_check.py` through `ie-ds41-protocol-fixture` (`partial` now also returns
  `cut_tool`): 6/6 -- prose, a cut inside a parameter, a cut before the name, a cut at the calls marker, a partial
  marker only, a second call cut (names the second).
- `~/ds41_work/p60/cut_e2e.py` on `ie serve`: a 160-token cap on "write a 60-line poem with write_file" ends
  `length` with `truncated_tool_call.name == write_file`, stream and non-stream, and no DSML in the content; a
  normal call (600 tokens) still returns `tool_calls` with no extra field.
