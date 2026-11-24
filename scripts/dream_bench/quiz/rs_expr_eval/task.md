Write `solution.rs` (a Rust module, no external crates) with:

```rust
#[derive(Debug, PartialEq, Eq)]
pub enum EvalError { Syntax, DivByZero, Overflow }

pub fn eval(s: &str) -> Result<i64, EvalError>
```

`eval` computes an integer expression:

- Tokens: non-negative decimal integer literals (digits only, leading zeros allowed), `+ - * / %`, `(` and `)`. Whitespace (space, `\t`, `\n`, `\r` and form feed `\x0C`; not vertical tab) may appear between tokens; anything else is a syntax error. Digits separated by whitespace are two literals (`"1 2"` is a syntax error).
- `*`, `/` and `%` bind tighter than `+` and `-`; binary operators of the same level are left-associative. Unary `-` and unary `+` may prefix any operand, repeatedly (`"2--3"` is 5, `"-(2+3)"` is -5).
- `/` and `%` behave like Rust's `i64` operators (truncate toward zero; the remainder takes the sign of the left operand).
- The whole input is checked for syntax first: if it is not a valid expression (including empty input or unbalanced parentheses) the result is `Err(Syntax)`, whatever else is wrong with it.
- Otherwise it is evaluated left operand before right operand, and the first error met is returned: division or remainder by zero gives `Err(DivByZero)`; any result outside `i64` gives `Err(Overflow)`, including `i64::MIN / -1`, `i64::MIN % -1`, negating `i64::MIN`, and a literal above `i64::MAX` (even directly after a unary minus: `"-9223372036854775808"` is `Err(Overflow)`).
- It must handle at least 1,000 levels of nested parentheses or unary minus, and a chain of 100,000 additions.
