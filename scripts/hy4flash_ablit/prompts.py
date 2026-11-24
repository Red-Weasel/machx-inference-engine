#!/usr/bin/env python3
"""Hy4-flash abliteration, step 1: chat prompts -> token-id batch files for ie-hy4flash-run --batch.

  python prompts.py --ie-dir ~/models/Hy4-flash-ie --data <dir> --out <dir>
                    [--n-train 128] [--n-test-harmful 48] [--n-test-harmless 32] [--reasoning no_think]

<data> holds harmful_train.json, harmless_train.json, harmful_test.json, harmless_test.json (lists of instructions: the
harmful_behaviors / harmless_alpaca sets our earlier abliterations used). Every instruction becomes one user turn rendered
with the model's own chat template (generation prompt included, thinking off by default), tokenized with its tokenizer.
Writes <out>/{train,test}_{harmful,harmless}.tsv ("name<TAB>id id ...") and prompts.json (texts and settings).
Run with the reference venv (transformers at the Hy4-flash pull request).
"""
import argparse, json, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie-dir", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=128); ap.add_argument("--n-test-harmful", type=int, default=48)
    ap.add_argument("--n-test-harmless", type=int, default=32); ap.add_argument("--reasoning", default="no_think")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.expanduser(a.ie_dir))
    os.makedirs(a.out, exist_ok=True)
    meta = {"reasoning_effort": a.reasoning, "sets": {}}
    def render(text):
        s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, reasoning_effort=a.reasoning)
        return s, tok(s, add_special_tokens=False)["input_ids"]
    for split, kind, n in (("train", "harmful", a.n_train), ("train", "harmless", a.n_train),
                           ("test", "harmful", a.n_test_harmful), ("test", "harmless", a.n_test_harmless)):
        texts = json.load(open(os.path.join(a.data, f"{kind}_{split}.json")))[:n]
        lens = []
        with open(os.path.join(a.out, f"{split}_{kind}.tsv"), "w") as f:
            for i, t in enumerate(texts):
                s, ids = render(t); lens.append(len(ids))
                f.write(f"{kind}_{split}_{i}\t{' '.join(map(str, ids))}\n")
        meta["sets"][f"{split}_{kind}"] = {"n": len(texts), "texts": texts, "tokens_min": min(lens), "tokens_max": max(lens)}
        print(f"{split}_{kind}: {len(texts)} prompts, {min(lens)}..{max(lens)} tokens")
    s, ids = render("Hello")
    meta["example"] = {"text": s, "ids": ids}
    json.dump(meta, open(os.path.join(a.out, "prompts.json"), "w"), indent=1, ensure_ascii=False)
    print("example prompt:", repr(s)); print("ids:", ids)

if __name__ == "__main__": main()
