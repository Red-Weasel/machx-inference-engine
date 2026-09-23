#!/usr/bin/env python3
"""P6.2 gate clause (b) inputs (docs/mimo26/00_PORT_PLAN.md): teacher-forced image sequences for ie-mimo26-score --image.
  make_p62_ids.py <model_dir> <answers.json> <out ids.txt>
answers.json (tools/mimo26/vision_serve_test.py --out): {"0": {"content": ...}, "1": ..., "2": ...} -- the engine's own
greedy answers to QUESTION on fixture_k.png. Each sequence = the checkpoint's Jinja template over
[user: image part + QUESTION] with the generation prompt (thinking off), tokenized with tokenizer.json (ONE
<|image_pad|> id per image: the plain tokenizer does not expand it; ie-mimo26-score --image does, as the processor
would), followed by the answer's tokens. The question text must equal vision_serve_test.py's.
"""
import json
import os
import sys

from transformers import PreTrainedTokenizerFast

model, answers_path, out = sys.argv[1:4]
QUESTION = "Describe this image in one sentence: the shapes, their colours, and any text."
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model, "tokenizer.json"))
tmpl = open(os.path.join(model, "chat_template.jinja"), encoding="utf-8").read()
answers = json.load(open(answers_path, encoding="utf-8"))
PAD = 151655
with open(out, "w") as f:
    for k in sorted(answers, key=int):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False, chat_template=tmpl)
        ids = tok.encode(prompt, add_special_tokens=False)
        assert ids.count(PAD) == 1, f"fixture {k}: {ids.count(PAD)} image pads in the prompt"
        ans = tok.encode(answers[k]["content"], add_special_tokens=False)
        seq = ids + ans
        f.write(" ".join(map(str, seq)) + "\n")
        print(f"fixture {k}: {len(ids)} prompt ids (pad at {ids.index(PAD)}) + {len(ans)} answer ids")
print(f"-> {out}")
