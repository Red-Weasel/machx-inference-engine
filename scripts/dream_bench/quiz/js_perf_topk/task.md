Write `solution.mjs` (an ES module) exporting `topKWords(text, k)`.

- A word is a maximal run of ASCII letters `a`-`z`, after lowercasing the text. Everything else separates words (`"don't"` is the two words `don` and `t`; digits and accented letters are separators).
- Return the `k` most frequent words as `[word, count]` pairs, sorted by count descending, ties by word in ascending code-unit order.
- If there are fewer than `k` distinct words, return them all. If `k` is not a positive number (including any value whose type is not `number`, such as the string `"2"`), return `[]`. A non-integer `k` is rounded down.
- A 5 MB text must take well under a second and a half.
