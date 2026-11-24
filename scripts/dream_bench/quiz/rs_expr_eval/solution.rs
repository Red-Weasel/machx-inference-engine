#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum EvalError { Syntax, DivByZero, Overflow }

use EvalError::*;

struct P<'a> { b: &'a [u8], i: usize, ev: bool }

impl<'a> P<'a> {
    fn peek(&mut self) -> Option<u8> {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() { self.i += 1; }
        self.b.get(self.i).copied()
    }
    fn expr(&mut self) -> Result<i64, EvalError> {
        let mut v = self.term()?;
        loop {
            match self.peek() {
                Some(c @ (b'+' | b'-')) => {
                    self.i += 1;
                    let r = self.term()?;
                    if self.ev { v = if c == b'+' { v.checked_add(r) } else { v.checked_sub(r) }.ok_or(Overflow)?; }
                }
                _ => return Ok(v),
            }
        }
    }
    fn term(&mut self) -> Result<i64, EvalError> {
        let mut v = self.unary()?;
        loop {
            match self.peek() {
                Some(c @ (b'*' | b'/' | b'%')) => {
                    self.i += 1;
                    let r = self.unary()?;
                    if self.ev {
                        v = match c {
                            b'*' => v.checked_mul(r).ok_or(Overflow)?,
                            _ if r == 0 => return Err(DivByZero),
                            b'/' => v.checked_div(r).ok_or(Overflow)?,
                            _ => v.checked_rem(r).ok_or(Overflow)?,
                        };
                    }
                }
                _ => return Ok(v),
            }
        }
    }
    fn unary(&mut self) -> Result<i64, EvalError> {
        match self.peek() {
            Some(b'-') => { self.i += 1; let v = self.unary()?; if self.ev { v.checked_neg().ok_or(Overflow) } else { Ok(0) } }
            Some(b'+') => { self.i += 1; self.unary() }
            _ => self.primary(),
        }
    }
    fn primary(&mut self) -> Result<i64, EvalError> {
        match self.peek() {
            Some(b'(') => {
                self.i += 1;
                let v = self.expr()?;
                if self.peek() != Some(b')') { return Err(Syntax); }
                self.i += 1;
                Ok(v)
            }
            Some(c) if c.is_ascii_digit() => {
                let st = self.i;
                while self.i < self.b.len() && self.b[self.i].is_ascii_digit() { self.i += 1; }
                if !self.ev { return Ok(0); }
                std::str::from_utf8(&self.b[st..self.i]).unwrap().parse::<i64>().map_err(|_| Overflow)
            }
            _ => Err(Syntax),
        }
    }
}

pub fn eval(s: &str) -> Result<i64, EvalError> {
    let mut chk = P { b: s.as_bytes(), i: 0, ev: false };
    chk.expr()?;
    if chk.peek().is_some() { return Err(Syntax); }
    P { b: s.as_bytes(), i: 0, ev: true }.expr()
}
