Write `solution.mjs` (an ES module) exporting `parseCSV(text, { delimiter = ',' } = {})`, returning an array of rows, each an array of strings.

- Records end at `\r\n`, `\n`, or a lone `\r`. A line break at the very end of the text does not start an extra empty row. An empty text gives `[]`. An empty line in the middle is a row with one empty field: `"a\n\nb"` -> `[["a"], [""], ["b"]]`.
- A field that starts with `"` is quoted: it runs to the next `"` that is not doubled; inside it, `""` is one `"`, and delimiters and line breaks are ordinary characters. After the closing quote only a delimiter, a line break or the end of the text may follow, otherwise throw a `SyntaxError`. A quote that never closes throws a `SyntaxError`.
- In an unquoted field a `"` is an ordinary character (`a"b` -> `a"b`). Fields are not trimmed.
- `"a,"` -> `[["a", ""]]`; `","` -> `[["", ""]]`.
- `delimiter` must be a single character other than `"`, `\r` and `\n`, otherwise throw a `TypeError`. A `text` that is not a string throws a `TypeError`.
