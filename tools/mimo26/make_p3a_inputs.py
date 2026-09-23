#!/usr/bin/env python3
"""P3a inputs (docs/mimo26/00_PORT_PLAN.md): token-id files for gate clauses (a)/(b) and needle prompts for (e).

  make_p3a_inputs.py <model_dir> <out_dir>

Writes: seq3000.txt (the first 3000 wikitext-2 test tokens, one line), seq4096.txt (the next 4096), and
needle_{32k,120k}_{10,50,90}.txt -- chat-template prompts (thinking off) whose user turn is wikitext filler with one
sentence "The secret passphrase is <P>." inserted at 10 / 50 / 90 % of the filler, followed by the question; the
passphrase for each is in needles.json.
"""
import json
import os
import sys

from tokenizers import Tokenizer

model, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
tok = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
wiki = open(os.path.expanduser("~/llama.cpp/wikitext-2-raw/wiki.test.raw"), encoding="utf-8").read()
ids = tok.encode(wiki, add_special_tokens=False).ids
open(os.path.join(out, "seq3000.txt"), "w").write(" ".join(map(str, ids[:3000])) + "\n")
open(os.path.join(out, "seq4096.txt"), "w").write(" ".join(map(str, ids[3000:3000 + 4096])) + "\n")

# the filler: wikitext repeated as needed (the train split is not on disk; test repeats past ~300k tokens)
words = wiki.split(" ")
needles = {}
PASS = {"32k": ["VIOLET-HARBOR-7361", "COPPER-MEADOW-2048", "SILENT-RIVER-9152"],
        "120k": ["AMBER-GLACIER-4417", "NORTH-LANTERN-8830", "OAK-TEMPEST-1296"]}
for size, target in (("32k", 32000), ("120k", 120000)):
    # grow the filler until it tokenizes to ~target
    n_words = int(target * 0.72)
    while True:
        filler = " ".join((words * (1 + n_words // len(words)))[:n_words])
        n_tok = len(tok.encode(filler, add_special_tokens=False).ids)
        if abs(n_tok - target) < 400:
            break
        n_words = int(n_words * target / n_tok)
    for frac, pw in zip((10, 50, 90), PASS[size]):
        cut = int(len(filler) * frac / 100)
        cut = filler.rfind(" ", 0, cut)
        text = filler[:cut] + f" The secret passphrase is {pw}. " + filler[cut:]
        user = ("Read the following text carefully. Somewhere in it is a secret passphrase.\n\n" + text +
                "\n\nWhat is the secret passphrase stated in the text? Reply with the passphrase only.")
        prompt = "<|im_start|>user\n" + user + "<|im_end|><|im_start|>assistant\n<think></think>"
        name = f"needle_{size}_{frac}.txt"
        open(os.path.join(out, name), "w", encoding="utf-8").write(prompt)
        needles[name] = {"passphrase": pw, "tokens": len(tok.encode(prompt, add_special_tokens=False).ids)}
json.dump(needles, open(os.path.join(out, "needles.json"), "w"), indent=1)
print(json.dumps(needles, indent=1))
