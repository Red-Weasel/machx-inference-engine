#!/usr/bin/env python3
"""Tokenizer goldens for DeepSeek-V4.1-Flash (docs/deepseek41/31, Phase 12 step 1).

Writes, for each text, <out>/tok_<name>.txt (UTF-8) and <out>/tok_<name>.i32 (int32 LE ids) from
the checkpoint's own tokenizer.json through HF `tokenizers`:
  docs    the repo's docs/*.md concatenated (the held-out corpus' source), the first ~300 KB
  stress  a hand-written mix: digits, CJK, code, punctuation, emoji, case, whitespace, specials
  pp2048  the pp_ids_2048.i32 text (decoded here) re-encoded: a round trip the engine must match
Usage: tokenizer_golden.py <model_dir> <docs_dir> <golden_dir> (needs the venv with tokenizers)"""
import glob, os, struct, sys
from tokenizers import Tokenizer

model, docs, out = sys.argv[1:4]
tok = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))

def write(name, text):
    ids = tok.encode(text, add_special_tokens=True).ids
    with open(os.path.join(out, "tok_%s.txt" % name), "w", encoding="utf-8", newline="") as f: f.write(text)
    with open(os.path.join(out, "tok_%s.i32" % name), "wb") as f: f.write(struct.pack("<%di" % len(ids), *ids))
    back = tok.decode(ids, skip_special_tokens=False)
    print("%-8s %8d chars %8d ids  round-trip %s" % (name, len(text), len(ids), "exact" if back == text else "DIFFERS (reference itself)"))
    return ids

parts = []
for p in sorted(glob.glob(os.path.join(docs, "*.md"))):
    with open(p, encoding="utf-8", errors="replace") as f: parts.append(f.read())
corpus = "\n".join(parts)
cut = corpus.encode("utf-8")[:300000].decode("utf-8", errors="ignore")
write("docs", cut)

stress = ("Numbers: 0 7 42 1234567 3.14159 2026-09-13 12:34:56 1,000,000 0x7F 1e-9 -273.15\n"
          "CJK: 深度求索 東京タワー こんにちは、世界。 カタカナ テスト 漢字とひらがなの混在 中文 English mixed 混合文本 123\n"
          "Code: def f(x): return x**2  # comment\n\tif a<=b and c!=d: pass\n    print(f\"{x!r} {y:>8.3f}\")\n"
          "  int main(void){return 0;} // C\n  SELECT * FROM t WHERE id IN (1,2,3);\n"
          "Case: iPhone HTTPServer XMLHttpRequest macOS NASA's UNIX-like eBay ALLCAPS lowercase Title Case\n"
          "Punct: ... !!! ??? -- --- (a) [b] {c} <d> \"quoted\" 'single' `tick` ~tilde~ @at #hash $5 %pct ^caret &amp; *star +plus =eq |pipe \\back /slash\n"
          "Emoji: 🚀🔥 😀 👍🏽 🇺🇸 ❤️ ✨ text🎉more\n"
          "Whitespace:   three   spaces\t\ttabs\n\n\nnewlines   \n trailing  \n"
          "Specials: <｜User｜>Hello there<｜Assistant｜>Hi! <｜begin▁of▁sentence｜> <｜end▁of▁sentence｜> <｜System｜>sys<｜DSML｜ calls>\n"
          "Words: the quick brown fox jumps over the lazy dog. The QUICK brown Fox. don't can't won't it's I'm you're they've\n"
          "Long token run: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz\n"
          "URLs: https://example.com/path?q=1&r=2#frag user@example.com 192.168.0.1 ::1\n"
          "Math: ∑_{i=1}^{n} x_i² ≈ π·r² ≠ ∞ → ← ⇔ ∀x∈ℝ α β γ δ\n"
          "Mixed script: Привет мир Γειά σου κόσμε مرحبا بالعالم שלום עולם नमस्ते दुनिया 안녕하세요 세계\n")
write("stress", stress)

pp = os.path.join(out, "pp_ids_2048.i32")
if os.path.exists(pp):
    with open(pp, "rb") as f: ids = list(struct.unpack("<2048i", f.read(2048 * 4)))
    text = tok.decode(ids, skip_special_tokens=False)
    re_ids = write("pp2048", text)
    print("pp2048: re-encoded ids equal the original %s" % ("yes" if re_ids == ids else "NO (%d/%d)" % (sum(a != b for a, b in zip(re_ids, ids)), len(ids))))
print("vocab size", tok.get_vocab_size(with_added_tokens=True))
