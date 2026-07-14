//! Expression language for blueprint `compute:` primitives.
//!
//! Tiny row-level + grouped-aggregation language. Used by `derive`,
//! `filter`, and `aggregate` primitives. Hand-rolled Pratt parser
//! (~200 LoC); tree-walking evaluator (~150 LoC). Fixed function
//! vocabulary — no user-defined functions, no graph traversal,
//! no SQL JOIN clauses. Keeps the surface stable across releases.
//!
//! ```text
//! expr      := or_expr
//! or_expr   := and_expr ("||" and_expr)*
//! and_expr  := not_expr ("&&" not_expr)*
//! not_expr  := "!" not_expr | cmp_expr
//! cmp_expr  := add_expr (cmp_op add_expr)?     -- chained cmps not allowed
//! cmp_op    := "==" | "!=" | "<" | "<=" | ">" | ">=" | "in"
//! add_expr  := mul_expr (("+" | "-") mul_expr)*
//! mul_expr  := unary    (("*" | "/" | "%") unary)*
//! unary     := "-" unary | primary
//! primary   := number | string | bool | "null"
//!            | ident
//!            | ident "(" args? ")"               -- function call
//!            | "[" args? "]"                     -- list literal
//!            | "(" expr ")"
//! args      := arg ("," arg)*
//! arg       := ident "=" expr | expr            -- named or positional
//! ```
//!
//! Function calls support an optional named-argument form (used by
//! the aggregate primitives, e.g. `last(shares_owned_after,
//! by=transaction_date)`). Named args are syntactic only — the
//! evaluator passes them through as `(Option<String>, Value)` pairs.

// K1 ships the expression engine standalone. Consumers (K3 derive/filter,
// K6 aggregate) wire up in subsequent phases; until then everything below
// is dead from the compiler's perspective.

use std::fmt;

// ─── Values ──────────────────────────────────────────────────────────────

/// Runtime value type. Split Int/Float so integer arithmetic stays
/// exact (no implicit float widening unless a Float operand forces it).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    List(Vec<Value>),
}

impl Value {
    /// Truthy semantics: bool(true), non-zero Int, non-zero non-NaN
    /// Float, non-empty String, non-empty List. Null + false-ish
    /// scalars are falsy.
    pub fn truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0 && !f.is_nan(),
            Value::String(s) => !s.is_empty(),
            Value::List(l) => !l.is_empty(),
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) => "string",
            Value::List(_) => "list",
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(x) => write!(f, "{}", x),
            Value::String(s) => write!(f, "{}", s),
            Value::List(l) => {
                write!(f, "[")?;
                for (i, v) in l.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
        }
    }
}

// ─── AST ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Value),
    /// Identifier reference — resolves against the row's `Bindings`.
    Ident(String),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    /// Function call. Each argument is `(name?, expr)` — name is
    /// populated for `kw=expr` arguments, `None` otherwise.
    Call(String, Vec<(Option<String>, Expr)>),
    List(Vec<Expr>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    In,
}

// ─── Errors ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ExprError {
    Parse(String),
    UndefinedIdent(String),
    UnknownFunction(String),
    WrongArgs {
        name: String,
        expected: String,
        got: usize,
    },
    TypeError(String),
    DivByZero,
    AggregateOutsideAgg(String),
}

impl fmt::Display for ExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprError::Parse(s) => write!(f, "parse: {}", s),
            ExprError::UndefinedIdent(s) => write!(f, "undefined identifier: {}", s),
            ExprError::UnknownFunction(s) => write!(f, "unknown function: {}", s),
            ExprError::WrongArgs {
                name,
                expected,
                got,
            } => {
                write!(f, "{}: expected {} args, got {}", name, expected, got)
            }
            ExprError::TypeError(s) => write!(f, "type error: {}", s),
            ExprError::DivByZero => write!(f, "division by zero"),
            ExprError::AggregateOutsideAgg(s) => {
                write!(
                    f,
                    "aggregate function '{}' used outside aggregate context",
                    s
                )
            }
        }
    }
}

impl std::error::Error for ExprError {}

pub type ExprResult<T> = Result<T, ExprError>;

// ─── Bindings (row-level identifier lookup) ──────────────────────────────

/// Per-row identifier→value lookup. Implementations: a HashMap
/// (testing), a typed column accessor (`derive` / `filter` primitives),
/// a row-with-group-state wrapper (`aggregate` primitive's per-row pass).
pub trait Bindings {
    fn get(&self, name: &str) -> Option<Value>;
}

impl Bindings for std::collections::HashMap<String, Value> {
    fn get(&self, name: &str) -> Option<Value> {
        self.get(name).cloned()
    }
}

// ─── Tokenizer ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Int(i64),
    Float(f64),
    Str(String),
    Ident(String),
    Bool(bool),
    Null,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Eq, // = (in named args / equality dispatch)
    Op(String),
}

fn tokenize(src: &str) -> ExprResult<Vec<Tok>> {
    let bytes = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
            // Handle scientific notation: 1e6, 1.5e-3
            if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                i += 1;
                if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                    i += 1;
                }
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let s = &src[start..i];
            if s.contains('.') || s.contains('e') || s.contains('E') {
                let v = s
                    .parse::<f64>()
                    .map_err(|e| ExprError::Parse(format!("bad number {}: {}", s, e)))?;
                out.push(Tok::Float(v));
            } else {
                let v = s
                    .parse::<i64>()
                    .map_err(|e| ExprError::Parse(format!("bad int {}: {}", s, e)))?;
                out.push(Tok::Int(v));
            }
            continue;
        }
        if c == b'"' || c == b'\'' {
            let quote = c;
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i] != quote {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if i >= bytes.len() {
                return Err(ExprError::Parse("unterminated string".to_string()));
            }
            let s = unescape(&src[start..i]);
            i += 1; // closing quote
            out.push(Tok::Str(s));
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let s = &src[start..i];
            match s {
                "true" => out.push(Tok::Bool(true)),
                "false" => out.push(Tok::Bool(false)),
                "null" => out.push(Tok::Null),
                _ => out.push(Tok::Ident(s.to_string())),
            }
            continue;
        }
        // Multi-char operators first.
        let two = if i + 1 < bytes.len() {
            &src[i..i + 2]
        } else {
            ""
        };
        match two {
            "==" | "!=" | "<=" | ">=" | "&&" | "||" => {
                out.push(Tok::Op(two.to_string()));
                i += 2;
                continue;
            }
            _ => {}
        }
        match c {
            b'(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            b')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            b'[' => {
                out.push(Tok::LBracket);
                i += 1;
            }
            b']' => {
                out.push(Tok::RBracket);
                i += 1;
            }
            b',' => {
                out.push(Tok::Comma);
                i += 1;
            }
            b'=' => {
                out.push(Tok::Eq);
                i += 1;
            }
            b'+' | b'-' | b'*' | b'/' | b'%' | b'<' | b'>' | b'!' => {
                out.push(Tok::Op((c as char).to_string()));
                i += 1;
            }
            _ => {
                return Err(ExprError::Parse(format!(
                    "unexpected character '{}'",
                    c as char
                )))
            }
        }
    }
    Ok(out)
}

fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('\'') => out.push('\''),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

// ─── Parser ──────────────────────────────────────────────────────────────

struct Parser {
    toks: Vec<Tok>,
    i: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i)
    }

    fn eat(&mut self) -> Option<Tok> {
        if self.i < self.toks.len() {
            let t = self.toks[self.i].clone();
            self.i += 1;
            Some(t)
        } else {
            None
        }
    }

    fn parse(&mut self) -> ExprResult<Expr> {
        let e = self.parse_or()?;
        if self.peek().is_some() {
            return Err(ExprError::Parse(format!(
                "unexpected trailing tokens: {:?}",
                &self.toks[self.i..]
            )));
        }
        Ok(e)
    }

    fn parse_or(&mut self) -> ExprResult<Expr> {
        let mut lhs = self.parse_and()?;
        while matches!(self.peek(), Some(Tok::Op(s)) if s == "||") {
            self.eat();
            let rhs = self.parse_and()?;
            lhs = Expr::Binary(BinaryOp::Or, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> ExprResult<Expr> {
        let mut lhs = self.parse_not()?;
        while matches!(self.peek(), Some(Tok::Op(s)) if s == "&&") {
            self.eat();
            let rhs = self.parse_not()?;
            lhs = Expr::Binary(BinaryOp::And, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_not(&mut self) -> ExprResult<Expr> {
        if matches!(self.peek(), Some(Tok::Op(s)) if s == "!") {
            self.eat();
            let rhs = self.parse_not()?;
            Ok(Expr::Unary(UnaryOp::Not, Box::new(rhs)))
        } else {
            self.parse_cmp()
        }
    }

    fn parse_cmp(&mut self) -> ExprResult<Expr> {
        let lhs = self.parse_add()?;
        let op = match self.peek() {
            Some(Tok::Op(s)) if matches!(s.as_str(), "==" | "!=" | "<" | "<=" | ">" | ">=") => {
                Some(s.clone())
            }
            Some(Tok::Ident(s)) if s == "in" => Some("in".to_string()),
            _ => None,
        };
        let Some(op) = op else { return Ok(lhs) };
        self.eat();
        let rhs = self.parse_add()?;
        let binop = match op.as_str() {
            "==" => BinaryOp::Eq,
            "!=" => BinaryOp::Ne,
            "<" => BinaryOp::Lt,
            "<=" => BinaryOp::Le,
            ">" => BinaryOp::Gt,
            ">=" => BinaryOp::Ge,
            "in" => BinaryOp::In,
            _ => unreachable!(),
        };
        Ok(Expr::Binary(binop, Box::new(lhs), Box::new(rhs)))
    }

    fn parse_add(&mut self) -> ExprResult<Expr> {
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Op(s)) if s == "+" || s == "-" => s.clone(),
                _ => break,
            };
            self.eat();
            let rhs = self.parse_mul()?;
            let binop = if op == "+" {
                BinaryOp::Add
            } else {
                BinaryOp::Sub
            };
            lhs = Expr::Binary(binop, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_mul(&mut self) -> ExprResult<Expr> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Op(s)) if s == "*" || s == "/" || s == "%" => s.clone(),
                _ => break,
            };
            self.eat();
            let rhs = self.parse_unary()?;
            let binop = match op.as_str() {
                "*" => BinaryOp::Mul,
                "/" => BinaryOp::Div,
                "%" => BinaryOp::Mod,
                _ => unreachable!(),
            };
            lhs = Expr::Binary(binop, Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> ExprResult<Expr> {
        if matches!(self.peek(), Some(Tok::Op(s)) if s == "-") {
            self.eat();
            let rhs = self.parse_unary()?;
            Ok(Expr::Unary(UnaryOp::Neg, Box::new(rhs)))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> ExprResult<Expr> {
        let t = self
            .eat()
            .ok_or_else(|| ExprError::Parse("unexpected end of expression".to_string()))?;
        match t {
            Tok::Int(n) => Ok(Expr::Literal(Value::Int(n))),
            Tok::Float(x) => Ok(Expr::Literal(Value::Float(x))),
            Tok::Str(s) => Ok(Expr::Literal(Value::String(s))),
            Tok::Bool(b) => Ok(Expr::Literal(Value::Bool(b))),
            Tok::Null => Ok(Expr::Literal(Value::Null)),
            Tok::LParen => {
                let e = self.parse_or()?;
                match self.eat() {
                    Some(Tok::RParen) => Ok(e),
                    other => Err(ExprError::Parse(format!("expected ')', got {:?}", other))),
                }
            }
            Tok::LBracket => {
                let mut items = Vec::new();
                if !matches!(self.peek(), Some(Tok::RBracket)) {
                    loop {
                        items.push(self.parse_or()?);
                        match self.peek() {
                            Some(Tok::Comma) => {
                                self.eat();
                            }
                            Some(Tok::RBracket) => break,
                            other => {
                                return Err(ExprError::Parse(format!(
                                    "list: expected ',' or ']', got {:?}",
                                    other
                                )));
                            }
                        }
                    }
                }
                self.eat(); // ]
                Ok(Expr::List(items))
            }
            Tok::Ident(name) => {
                if matches!(self.peek(), Some(Tok::LParen)) {
                    self.eat(); // (
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Tok::RParen)) {
                        loop {
                            args.push(self.parse_arg()?);
                            match self.peek() {
                                Some(Tok::Comma) => {
                                    self.eat();
                                }
                                Some(Tok::RParen) => break,
                                other => {
                                    return Err(ExprError::Parse(format!(
                                        "call: expected ',' or ')', got {:?}",
                                        other
                                    )));
                                }
                            }
                        }
                    }
                    self.eat(); // )
                                // Special-case: count(*) → count() with a sentinel marker.
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(Expr::Ident(name))
                }
            }
            other => Err(ExprError::Parse(format!("unexpected token: {:?}", other))),
        }
    }

    /// Parse one function-call argument. Supports `name=expr` for
    /// named args and bare `expr` for positional. Also recognises the
    /// `count(*)` special form by emitting a sentinel star-ident.
    fn parse_arg(&mut self) -> ExprResult<(Option<String>, Expr)> {
        // count(*) — single-token star.
        if matches!(self.peek(), Some(Tok::Op(s)) if s == "*") {
            self.eat();
            return Ok((None, Expr::Ident("*".to_string())));
        }
        // Lookahead for `name=expr`.
        if let (Some(Tok::Ident(name)), Some(Tok::Eq)) =
            (self.toks.get(self.i), self.toks.get(self.i + 1))
        {
            let n = name.clone();
            self.i += 2;
            let v = self.parse_or()?;
            return Ok((Some(n), v));
        }
        let v = self.parse_or()?;
        Ok((None, v))
    }
}

/// Parse an expression source string into an AST.
pub fn parse(src: &str) -> ExprResult<Expr> {
    let toks = tokenize(src)?;
    let mut p = Parser { toks, i: 0 };
    p.parse()
}

// ─── Evaluator ───────────────────────────────────────────────────────────

/// Tree-walking evaluator. Row-level only — aggregate functions raise
/// an error here; they're dispatched separately by the `aggregate`
/// primitive in K6.
pub fn eval(expr: &Expr, ctx: &dyn Bindings) -> ExprResult<Value> {
    match expr {
        Expr::Literal(v) => Ok(v.clone()),
        Expr::Ident(name) => {
            if name == "*" {
                // Bare `*` only meaningful inside count(*) — bubble up
                // so the evaluator wrapper for aggregate can see it.
                return Err(ExprError::Parse(
                    "bare '*' only valid inside count(*)".to_string(),
                ));
            }
            ctx.get(name)
                .ok_or_else(|| ExprError::UndefinedIdent(name.clone()))
        }
        Expr::Unary(op, inner) => {
            let v = eval(inner, ctx)?;
            apply_unary(*op, v)
        }
        Expr::Binary(op, lhs, rhs) => {
            // Short-circuit logical ops.
            match op {
                BinaryOp::And => {
                    let l = eval(lhs, ctx)?;
                    if !l.truthy() {
                        return Ok(Value::Bool(false));
                    }
                    let r = eval(rhs, ctx)?;
                    Ok(Value::Bool(r.truthy()))
                }
                BinaryOp::Or => {
                    let l = eval(lhs, ctx)?;
                    if l.truthy() {
                        return Ok(Value::Bool(true));
                    }
                    let r = eval(rhs, ctx)?;
                    Ok(Value::Bool(r.truthy()))
                }
                _ => {
                    let l = eval(lhs, ctx)?;
                    let r = eval(rhs, ctx)?;
                    apply_binary(*op, l, r)
                }
            }
        }
        Expr::Call(name, args) => {
            if is_aggregate_fn(name) {
                return Err(ExprError::AggregateOutsideAgg(name.clone()));
            }
            let mut vals = Vec::with_capacity(args.len());
            for (_kw, e) in args {
                vals.push(eval(e, ctx)?);
            }
            call_builtin(name, &vals)
        }
        Expr::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for e in items {
                out.push(eval(e, ctx)?);
            }
            Ok(Value::List(out))
        }
    }
}

fn apply_unary(op: UnaryOp, v: Value) -> ExprResult<Value> {
    match (op, v) {
        (UnaryOp::Neg, Value::Int(i)) => Ok(Value::Int(-i)),
        (UnaryOp::Neg, Value::Float(f)) => Ok(Value::Float(-f)),
        (UnaryOp::Neg, other) => Err(ExprError::TypeError(format!(
            "negate: cannot negate {}",
            other.type_name()
        ))),
        (UnaryOp::Not, v) => Ok(Value::Bool(!v.truthy())),
    }
}

fn apply_binary(op: BinaryOp, l: Value, r: Value) -> ExprResult<Value> {
    use BinaryOp::*;
    match op {
        Add | Sub | Mul | Div | Mod => apply_arith(op, l, r),
        Eq => Ok(Value::Bool(values_equal(&l, &r))),
        Ne => Ok(Value::Bool(!values_equal(&l, &r))),
        Lt | Le | Gt | Ge => apply_cmp(op, l, r),
        In => match r {
            Value::List(items) => Ok(Value::Bool(items.iter().any(|it| values_equal(it, &l)))),
            other => Err(ExprError::TypeError(format!(
                "in: expected list on right, got {}",
                other.type_name()
            ))),
        },
        And | Or => unreachable!("short-circuited above"),
    }
}

fn apply_arith(op: BinaryOp, l: Value, r: Value) -> ExprResult<Value> {
    use BinaryOp::*;
    // SQL-style null propagation: any operand null → null result.
    // Real-world CSV data (e.g. SEC insider grants with no price)
    // routinely has nulls — erroring would force `coalesce(x, 0)`
    // wrapping on every arithmetic expression, which is verbose
    // and obscures intent. Aggregate functions (sum/avg) already
    // skip nulls, so propagation composes cleanly.
    if matches!(l, Value::Null) || matches!(r, Value::Null) {
        return Ok(Value::Null);
    }
    // String concatenation for Add only.
    if op == Add {
        if let (Value::String(a), Value::String(b)) = (&l, &r) {
            return Ok(Value::String(format!("{}{}", a, b)));
        }
    }
    // Promote int/float.
    let (la, ra, is_float) = match (&l, &r) {
        (Value::Int(_), Value::Int(_)) => (l, r, false),
        (Value::Float(_), Value::Float(_)) => (l, r, true),
        (Value::Int(_), Value::Float(_)) | (Value::Float(_), Value::Int(_)) => (
            Value::Float(to_float(&l)?),
            Value::Float(to_float(&r)?),
            true,
        ),
        (a, b) => {
            return Err(ExprError::TypeError(format!(
                "{:?}: incompatible types {} and {}",
                op,
                a.type_name(),
                b.type_name()
            )));
        }
    };
    if is_float {
        let (a, b) = (to_float(&la)?, to_float(&ra)?);
        let v = match op {
            Add => a + b,
            Sub => a - b,
            Mul => a * b,
            Div => {
                if b == 0.0 {
                    return Err(ExprError::DivByZero);
                }
                a / b
            }
            Mod => a % b,
            _ => unreachable!(),
        };
        Ok(Value::Float(v))
    } else {
        let (a, b) = (to_int(&la)?, to_int(&ra)?);
        let v = match op {
            Add => a.checked_add(b),
            Sub => a.checked_sub(b),
            Mul => a.checked_mul(b),
            Div => {
                if b == 0 {
                    return Err(ExprError::DivByZero);
                }
                a.checked_div(b)
            }
            Mod => {
                if b == 0 {
                    return Err(ExprError::DivByZero);
                }
                a.checked_rem(b)
            }
            _ => unreachable!(),
        }
        .ok_or_else(|| ExprError::TypeError("integer overflow".to_string()))?;
        Ok(Value::Int(v))
    }
}

fn apply_cmp(op: BinaryOp, l: Value, r: Value) -> ExprResult<Value> {
    use BinaryOp::*;
    // SQL-style: null compared with anything yields null. In a
    // filter predicate, null is treated as false (so the row is
    // dropped) — handled by the truthy() conversion at the call
    // site.
    if matches!(l, Value::Null) || matches!(r, Value::Null) {
        return Ok(Value::Null);
    }
    let ord = match (&l, &r) {
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a
            .partial_cmp(b)
            .ok_or_else(|| ExprError::TypeError("NaN comparison".to_string()))?,
        (Value::Int(_), Value::Float(_)) | (Value::Float(_), Value::Int(_)) => to_float(&l)?
            .partial_cmp(&to_float(&r)?)
            .ok_or_else(|| ExprError::TypeError("NaN comparison".to_string()))?,
        (Value::String(a), Value::String(b)) => a.cmp(b),
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        (a, b) => {
            return Err(ExprError::TypeError(format!(
                "cmp: incompatible types {} and {}",
                a.type_name(),
                b.type_name()
            )));
        }
    };
    let v = match op {
        Lt => ord.is_lt(),
        Le => ord.is_le(),
        Gt => ord.is_gt(),
        Ge => ord.is_ge(),
        _ => unreachable!(),
    };
    Ok(Value::Bool(v))
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x == y,
        (Value::Int(x), Value::Float(y)) | (Value::Float(y), Value::Int(x)) => (*x as f64) == *y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::List(x), Value::List(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(a, b)| values_equal(a, b))
        }
        _ => false,
    }
}

fn to_int(v: &Value) -> ExprResult<i64> {
    match v {
        Value::Int(i) => Ok(*i),
        Value::Float(f) => Ok(*f as i64),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        other => Err(ExprError::TypeError(format!(
            "expected int, got {}",
            other.type_name()
        ))),
    }
}

fn to_float(v: &Value) -> ExprResult<f64> {
    match v {
        Value::Int(i) => Ok(*i as f64),
        Value::Float(f) => Ok(*f),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        other => Err(ExprError::TypeError(format!(
            "expected float, got {}",
            other.type_name()
        ))),
    }
}

// ─── Built-in functions ──────────────────────────────────────────────────

/// True for aggregate-only function names. `min` / `max` work in
/// BOTH row-level (variadic `min(a, b, c)`) and aggregate
/// (single-column `min(price)` across rows) contexts — they're not
/// listed here. The aggregate primitive (K6) handles the dispatch
/// based on argument shape.
pub fn is_aggregate_fn(name: &str) -> bool {
    matches!(
        name,
        "count" | "count_distinct" | "sum" | "avg" | "first" | "last"
    )
}

fn one<'a>(name: &str, args: &'a [Value]) -> ExprResult<&'a Value> {
    if args.len() != 1 {
        Err(ExprError::WrongArgs {
            name: name.to_string(),
            expected: "1".to_string(),
            got: args.len(),
        })
    } else {
        Ok(&args[0])
    }
}

fn call_builtin(name: &str, args: &[Value]) -> ExprResult<Value> {
    match name {
        // Math
        "abs" => match one(name, args)? {
            Value::Int(i) => Ok(Value::Int(i.abs())),
            Value::Float(f) => Ok(Value::Float(f.abs())),
            other => Err(ExprError::TypeError(format!(
                "abs: expected number, got {}",
                other.type_name()
            ))),
        },
        "round" => match args.len() {
            1 => Ok(Value::Float(to_float(&args[0])?.round())),
            2 => {
                let x = to_float(&args[0])?;
                let n = to_int(&args[1])?;
                let p = 10f64.powi(n as i32);
                Ok(Value::Float((x * p).round() / p))
            }
            n => Err(ExprError::WrongArgs {
                name: name.to_string(),
                expected: "1 or 2".to_string(),
                got: n,
            }),
        },
        "ceil" => Ok(Value::Float(to_float(one(name, args)?)?.ceil())),
        "floor" => Ok(Value::Float(to_float(one(name, args)?)?.floor())),
        "sqrt" => Ok(Value::Float(to_float(one(name, args)?)?.sqrt())),
        "exp" => Ok(Value::Float(to_float(one(name, args)?)?.exp())),
        "log" => match args.len() {
            1 => Ok(Value::Float(to_float(&args[0])?.ln())),
            2 => {
                let x = to_float(&args[0])?;
                let base = to_float(&args[1])?;
                Ok(Value::Float(x.log(base)))
            }
            n => Err(ExprError::WrongArgs {
                name: name.to_string(),
                expected: "1 or 2".to_string(),
                got: n,
            }),
        },
        "pow" => {
            if args.len() != 2 {
                return Err(ExprError::WrongArgs {
                    name: name.to_string(),
                    expected: "2".to_string(),
                    got: args.len(),
                });
            }
            Ok(Value::Float(to_float(&args[0])?.powf(to_float(&args[1])?)))
        }
        "min" => fold_min_max(name, args, true),
        "max" => fold_min_max(name, args, false),
        // Conditional
        "if" => {
            if args.len() != 3 {
                return Err(ExprError::WrongArgs {
                    name: name.to_string(),
                    expected: "3".to_string(),
                    got: args.len(),
                });
            }
            Ok(if args[0].truthy() {
                args[1].clone()
            } else {
                args[2].clone()
            })
        }
        "coalesce" => {
            for v in args {
                if !matches!(v, Value::Null) {
                    return Ok(v.clone());
                }
            }
            Ok(Value::Null)
        }
        // Type conversion
        "int" => Ok(Value::Int(to_int(one(name, args)?)?)),
        "float" => Ok(Value::Float(to_float(one(name, args)?)?)),
        "string" => Ok(Value::String(format!("{}", one(name, args)?))),
        // String
        "concat" => {
            let mut s = String::new();
            for v in args {
                s.push_str(&format!("{}", v));
            }
            Ok(Value::String(s))
        }
        "lower" => match one(name, args)? {
            Value::String(s) => Ok(Value::String(s.to_lowercase())),
            other => Err(ExprError::TypeError(format!(
                "lower: expected string, got {}",
                other.type_name()
            ))),
        },
        "upper" => match one(name, args)? {
            Value::String(s) => Ok(Value::String(s.to_uppercase())),
            other => Err(ExprError::TypeError(format!(
                "upper: expected string, got {}",
                other.type_name()
            ))),
        },
        "len" => match one(name, args)? {
            Value::String(s) => Ok(Value::Int(s.chars().count() as i64)),
            Value::List(l) => Ok(Value::Int(l.len() as i64)),
            other => Err(ExprError::TypeError(format!(
                "len: expected string or list, got {}",
                other.type_name()
            ))),
        },
        "contains" => {
            if args.len() != 2 {
                return Err(ExprError::WrongArgs {
                    name: name.to_string(),
                    expected: "2".to_string(),
                    got: args.len(),
                });
            }
            match (&args[0], &args[1]) {
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a.contains(b.as_str()))),
                _ => Err(ExprError::TypeError(
                    "contains: expected two strings".to_string(),
                )),
            }
        }
        "starts_with" => match (args.first(), args.get(1)) {
            (Some(Value::String(a)), Some(Value::String(b))) => {
                Ok(Value::Bool(a.starts_with(b.as_str())))
            }
            _ => Err(ExprError::TypeError(
                "starts_with: expected two strings".to_string(),
            )),
        },
        "ends_with" => match (args.first(), args.get(1)) {
            (Some(Value::String(a)), Some(Value::String(b))) => {
                Ok(Value::Bool(a.ends_with(b.as_str())))
            }
            _ => Err(ExprError::TypeError(
                "ends_with: expected two strings".to_string(),
            )),
        },
        // Date helpers (operate on YYYY-MM-DD strings; no chrono dep)
        "year" => Ok(Value::Int(parse_date_field(one(name, args)?, 0..4)?)),
        "month" => Ok(Value::Int(parse_date_field(one(name, args)?, 5..7)?)),
        "day" => Ok(Value::Int(parse_date_field(one(name, args)?, 8..10)?)),
        "quarter" => {
            let m = parse_date_field(one(name, args)?, 5..7)?;
            Ok(Value::Int(((m - 1) / 3) + 1))
        }
        _ => Err(ExprError::UnknownFunction(name.to_string())),
    }
}

fn fold_min_max(name: &str, args: &[Value], want_min: bool) -> ExprResult<Value> {
    if args.is_empty() {
        return Err(ExprError::WrongArgs {
            name: name.to_string(),
            expected: "≥1".to_string(),
            got: 0,
        });
    }
    let mut best = args[0].clone();
    for v in &args[1..] {
        let ord = match (&best, v) {
            (Value::Int(a), Value::Int(b)) => a.cmp(b),
            (Value::Float(a), Value::Float(b)) => a
                .partial_cmp(b)
                .ok_or_else(|| ExprError::TypeError(format!("{}: NaN", name)))?,
            (Value::Int(_), Value::Float(_)) | (Value::Float(_), Value::Int(_)) => to_float(&best)?
                .partial_cmp(&to_float(v)?)
                .ok_or_else(|| ExprError::TypeError(format!("{}: NaN", name)))?,
            (Value::String(a), Value::String(b)) => a.cmp(b),
            _ => {
                return Err(ExprError::TypeError(format!(
                    "{}: incompatible types",
                    name
                )))
            }
        };
        let take = if want_min { ord.is_gt() } else { ord.is_lt() };
        if take {
            best = v.clone();
        }
    }
    Ok(best)
}

fn parse_date_field(v: &Value, range: std::ops::Range<usize>) -> ExprResult<i64> {
    let s = match v {
        Value::String(s) => s,
        other => {
            return Err(ExprError::TypeError(format!(
                "date function: expected string YYYY-MM-DD, got {}",
                other.type_name()
            )))
        }
    };
    let slice = s
        .get(range.clone())
        .ok_or_else(|| ExprError::TypeError(format!("date too short: {}", s)))?;
    slice
        .parse::<i64>()
        .map_err(|_| ExprError::TypeError(format!("invalid date field in '{}'", s)))
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn ctx(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn eval_str(src: &str, bindings: &HashMap<String, Value>) -> ExprResult<Value> {
        let e = parse(src)?;
        eval(&e, bindings)
    }

    // ─── tokenizer ─────────────────────────────────────────────────

    #[test]
    fn tokenize_basics() {
        assert_eq!(tokenize("42").unwrap(), vec![Tok::Int(42)]);
        assert_eq!(tokenize("3.14").unwrap(), vec![Tok::Float(3.14)]);
        assert_eq!(tokenize("1e6").unwrap(), vec![Tok::Float(1e6)]);
        assert_eq!(
            tokenize("\"hi\"").unwrap(),
            vec![Tok::Str("hi".to_string())]
        );
        assert_eq!(tokenize("true").unwrap(), vec![Tok::Bool(true)]);
        assert_eq!(tokenize("null").unwrap(), vec![Tok::Null]);
        assert_eq!(
            tokenize("x + y").unwrap(),
            vec![
                Tok::Ident("x".to_string()),
                Tok::Op("+".to_string()),
                Tok::Ident("y".to_string())
            ]
        );
        assert_eq!(
            tokenize("a==b").unwrap(),
            vec![
                Tok::Ident("a".to_string()),
                Tok::Op("==".to_string()),
                Tok::Ident("b".to_string())
            ]
        );
    }

    #[test]
    fn tokenize_string_escape() {
        let t = tokenize(r#""hello\nworld""#).unwrap();
        assert_eq!(t, vec![Tok::Str("hello\nworld".to_string())]);
    }

    #[test]
    fn tokenize_unterminated_string_errors() {
        assert!(tokenize("\"oops").is_err());
    }

    // ─── arithmetic ────────────────────────────────────────────────

    #[test]
    fn arith_int() {
        let c = ctx(&[]);
        assert_eq!(eval_str("2 + 3", &c).unwrap(), Value::Int(5));
        assert_eq!(eval_str("2 * 3 + 1", &c).unwrap(), Value::Int(7));
        assert_eq!(eval_str("(2 + 3) * 4", &c).unwrap(), Value::Int(20));
        assert_eq!(eval_str("7 % 3", &c).unwrap(), Value::Int(1));
        assert_eq!(eval_str("-5", &c).unwrap(), Value::Int(-5));
    }

    #[test]
    fn arith_float() {
        let c = ctx(&[]);
        assert_eq!(eval_str("1.5 + 2.5", &c).unwrap(), Value::Float(4.0));
        assert_eq!(eval_str("10 / 4.0", &c).unwrap(), Value::Float(2.5));
    }

    #[test]
    fn arith_mixed_promotes() {
        let c = ctx(&[]);
        // int + float → float
        assert_eq!(eval_str("2 + 3.0", &c).unwrap(), Value::Float(5.0));
    }

    #[test]
    fn arith_div_zero_errors() {
        let c = ctx(&[]);
        assert_eq!(eval_str("5 / 0", &c).unwrap_err(), ExprError::DivByZero);
        assert_eq!(eval_str("5.0 / 0.0", &c).unwrap_err(), ExprError::DivByZero);
    }

    #[test]
    fn arith_int_overflow_errors() {
        let c = ctx(&[]);
        let src = format!("{} * 2", i64::MAX);
        assert!(eval_str(&src, &c).is_err());
    }

    // ─── comparison + logic ────────────────────────────────────────

    #[test]
    fn cmp_basic() {
        let c = ctx(&[]);
        assert_eq!(eval_str("5 > 3", &c).unwrap(), Value::Bool(true));
        assert_eq!(eval_str("5 <= 5", &c).unwrap(), Value::Bool(true));
        assert_eq!(eval_str("5 != 3", &c).unwrap(), Value::Bool(true));
        assert_eq!(eval_str("'a' < 'b'", &c).unwrap(), Value::Bool(true));
    }

    #[test]
    fn logic_short_circuit() {
        let c = ctx(&[("x", Value::Int(0))]);
        // x != 0 && (10 / x) > 0 — should short-circuit, NOT divide by zero
        assert_eq!(
            eval_str("x != 0 && 10 / x > 0", &c).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn logic_not() {
        let c = ctx(&[]);
        assert_eq!(eval_str("!true", &c).unwrap(), Value::Bool(false));
        assert_eq!(eval_str("!(1 == 1)", &c).unwrap(), Value::Bool(false));
    }

    // ─── membership ────────────────────────────────────────────────

    #[test]
    fn in_operator() {
        let c = ctx(&[("code", Value::String("P".to_string()))]);
        assert_eq!(
            eval_str("code in ['P', 'A']", &c).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval_str("'X' in ['P', 'A']", &c).unwrap(),
            Value::Bool(false)
        );
    }

    // ─── identifiers + bindings ────────────────────────────────────

    #[test]
    fn ident_lookup() {
        let c = ctx(&[("shares", Value::Int(100)), ("price", Value::Float(225.5))]);
        assert_eq!(
            eval_str("shares * price", &c).unwrap(),
            Value::Float(22550.0)
        );
    }

    #[test]
    fn undefined_ident_errors() {
        let c = ctx(&[]);
        assert!(matches!(
            eval_str("nope", &c),
            Err(ExprError::UndefinedIdent(_))
        ));
    }

    // ─── functions ─────────────────────────────────────────────────

    #[test]
    fn fn_abs() {
        let c = ctx(&[]);
        assert_eq!(eval_str("abs(-5)", &c).unwrap(), Value::Int(5));
        assert_eq!(eval_str("abs(-3.5)", &c).unwrap(), Value::Float(3.5));
    }

    #[test]
    fn fn_round() {
        let c = ctx(&[]);
        assert_eq!(eval_str("round(3.7)", &c).unwrap(), Value::Float(4.0));
        // 2 decimal places
        let v = eval_str("round(1.2345, 2)", &c).unwrap();
        if let Value::Float(f) = v {
            assert!((f - 1.23).abs() < 1e-9);
        } else {
            panic!("expected float");
        }
    }

    #[test]
    fn fn_if_coalesce() {
        let c = ctx(&[("x", Value::Int(10))]);
        assert_eq!(
            eval_str("if(x > 5, 'big', 'small')", &c).unwrap(),
            Value::String("big".to_string())
        );
        assert_eq!(
            eval_str("coalesce(null, null, 42)", &c).unwrap(),
            Value::Int(42)
        );
    }

    #[test]
    fn fn_type_casts() {
        let c = ctx(&[]);
        assert_eq!(eval_str("int(3.7)", &c).unwrap(), Value::Int(3));
        assert_eq!(eval_str("float(5)", &c).unwrap(), Value::Float(5.0));
        assert_eq!(
            eval_str("string(42)", &c).unwrap(),
            Value::String("42".to_string())
        );
    }

    #[test]
    fn fn_string_helpers() {
        let c = ctx(&[]);
        assert_eq!(
            eval_str("lower('ABC')", &c).unwrap(),
            Value::String("abc".to_string())
        );
        assert_eq!(
            eval_str("concat('a', 'b', 'c')", &c).unwrap(),
            Value::String("abc".to_string())
        );
        assert_eq!(
            eval_str("contains('hello world', 'world')", &c).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval_str("starts_with('hello', 'he')", &c).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(eval_str("len('abc')", &c).unwrap(), Value::Int(3));
    }

    #[test]
    fn fn_min_max() {
        let c = ctx(&[]);
        assert_eq!(eval_str("min(3, 1, 4, 1, 5)", &c).unwrap(), Value::Int(1));
        assert_eq!(eval_str("max(3, 1, 4, 1, 5)", &c).unwrap(), Value::Int(5));
    }

    #[test]
    fn fn_date() {
        let c = ctx(&[("d", Value::String("2025-11-14".to_string()))]);
        assert_eq!(eval_str("year(d)", &c).unwrap(), Value::Int(2025));
        assert_eq!(eval_str("month(d)", &c).unwrap(), Value::Int(11));
        assert_eq!(eval_str("day(d)", &c).unwrap(), Value::Int(14));
        assert_eq!(eval_str("quarter(d)", &c).unwrap(), Value::Int(4));
    }

    // ─── aggregate detection ───────────────────────────────────────

    #[test]
    fn aggregate_fn_errors_in_row_context() {
        let c = ctx(&[("x", Value::Int(5))]);
        let r = eval_str("sum(x)", &c);
        assert!(matches!(r, Err(ExprError::AggregateOutsideAgg(_))));
    }

    #[test]
    fn is_aggregate_fn_classifies_correctly() {
        assert!(is_aggregate_fn("sum"));
        assert!(is_aggregate_fn("count"));
        assert!(is_aggregate_fn("last"));
        assert!(!is_aggregate_fn("abs"));
        assert!(!is_aggregate_fn("if"));
    }

    // ─── named arg parsing ─────────────────────────────────────────

    #[test]
    fn parses_named_args() {
        let e = parse("last(shares, by=date)").unwrap();
        match e {
            Expr::Call(name, args) => {
                assert_eq!(name, "last");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0].0, None);
                assert_eq!(args[1].0, Some("by".to_string()));
            }
            _ => panic!("expected Call"),
        }
    }

    #[test]
    fn parses_count_star() {
        let e = parse("count(*)").unwrap();
        match e {
            Expr::Call(name, args) => {
                assert_eq!(name, "count");
                assert_eq!(args.len(), 1);
                match &args[0].1 {
                    Expr::Ident(s) => assert_eq!(s, "*"),
                    _ => panic!("expected star ident"),
                }
            }
            _ => panic!("expected Call"),
        }
    }

    // ─── precedence ────────────────────────────────────────────────

    #[test]
    fn precedence_correct() {
        let c = ctx(&[]);
        assert_eq!(eval_str("2 + 3 * 4", &c).unwrap(), Value::Int(14));
        assert_eq!(eval_str("(2 + 3) * 4", &c).unwrap(), Value::Int(20));
        assert_eq!(eval_str("!false && true", &c).unwrap(), Value::Bool(true));
        assert_eq!(
            eval_str("true || false && false", &c).unwrap(),
            Value::Bool(true)
        );
    }

    // ─── full SEC-style examples ───────────────────────────────────

    #[test]
    fn sec_total_value_expression() {
        let c = ctx(&[
            ("shares", Value::Float(100.0)),
            ("price_per_share", Value::Float(225.50)),
        ]);
        assert_eq!(
            eval_str("shares * price_per_share", &c).unwrap(),
            Value::Float(22550.0)
        );
    }

    #[test]
    fn sec_is_buy_predicate() {
        let buy = ctx(&[("transaction_code", Value::String("P".to_string()))]);
        let sell = ctx(&[("transaction_code", Value::String("S".to_string()))]);
        assert_eq!(
            eval_str("transaction_code in ['P', 'A']", &buy).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            eval_str("transaction_code in ['P', 'A']", &sell).unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn sec_unit_conversion() {
        let c = ctx(&[("value", Value::Float(12_500_000_000.0))]);
        let v = eval_str("value / 1e9", &c).unwrap();
        if let Value::Float(f) = v {
            assert!((f - 12.5).abs() < 1e-9);
        } else {
            panic!("expected float");
        }
    }
}
