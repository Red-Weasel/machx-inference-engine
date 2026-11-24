mod solution;
use solution::{eval, EvalError};
fn main() {
    let ok = |s: &str, v: i64| assert_eq!(eval(s), Ok(v), "{:?}", s);
    let er = |s: &str, e: EvalError| assert_eq!(eval(s), Err(e), "{:?}", s);
    ok("1+2*3", 7); ok("(1+2)*3", 9); ok("10-4-3", 3); ok("100/10/5", 2); ok("2*3%4", 2); ok("7/-2", -3); ok("-7/2", -3);
    ok("-7%3", -1); ok("7%-3", 1); ok(" \t2 *\n( 3+4 ) ", 14); ok("2--3", 5); ok("--3", 3); ok("+4", 4); ok("-(2+3)*2", -10);
    ok("2*-3", -6); ok("1-2*3+4", -1); ok("0", 0); ok("007", 7); ok("-2*-2", 4); ok("8/(3-1)/2", 2);
    ok("9223372036854775807", i64::MAX); ok("-9223372036854775807-1", i64::MIN);
    for s in ["", "   ", "1+", "(1", "1)", "()", "1 2", "2**3", "1+a", "1/0+", "\u{ff11}", "1..2", "(1+2", "1+(2*3))", "-", "(-)", "1 + + "] {
        er(s, EvalError::Syntax);
    }
    er("1/0", EvalError::DivByZero); er("5%(3-3)", EvalError::DivByZero); er("1/(2-2)*9223372036854775808", EvalError::DivByZero);
    er("9223372036854775808", EvalError::Overflow); er("-9223372036854775808", EvalError::Overflow);
    er("9223372036854775807+1", EvalError::Overflow); er("(-9223372036854775807-1)/-1", EvalError::Overflow);
    er("(-9223372036854775807-1)%-1", EvalError::Overflow); er("-(-9223372036854775807-1)", EvalError::Overflow);
    er("3037000500*3037000500", EvalError::Overflow); er("0-9223372036854775807-2", EvalError::Overflow);
    er("1/0 + 9223372036854775808", EvalError::DivByZero); er("9223372036854775808 + 1/0", EvalError::Overflow);
    let deep = "(".repeat(1000) + "1" + &")".repeat(1000); ok(&deep, 1);
    let negs = "-".repeat(999) + "5"; ok(&negs, -5);
    ok("1\r+\x0c2", 3); er("1\u{b}+2", EvalError::Syntax);
    let long = vec!["1"; 100_000].join("+"); ok(&long, 100_000);
    let long2 = vec!["2"; 50_000].join("*0+"); ok(&long2, 2);
    println!("{}", std::env::args().nth(1).unwrap_or_default());
}
