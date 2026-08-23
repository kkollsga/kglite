#[derive(Debug, Clone, PartialEq)]
pub enum CypherToken {
    // Keywords (case-insensitive)
    Match,
    Optional,
    Where,
    Return,
    With,
    Order,
    By,
    As,
    And,
    Or,
    Not,
    In,
    Is,
    Null,
    /// `NULLS` keyword used in ORDER BY clauses (e.g. `ORDER BY x DESC NULLS LAST`).
    /// 0.9.0 §2 — distinct from `Null`.
    Nulls,
    Limit,
    Skip,
    Unwind,
    Union,
    Intersect,
    Except,
    All,
    Distinct,
    Create,
    Set,
    Delete,
    Detach,
    Merge,
    Remove,
    Foreach,
    On,
    Asc,
    Desc,
    StartsWith,
    EndsWith,
    Contains,
    Case,
    When,
    Then,
    Else,
    End,
    True,
    False,
    Exists,
    Explain,
    Profile,
    Call,
    Yield,
    Over,
    Partition,
    Having,
    Xor,

    // Both spellings produce this one token — they are the same reference to
    // the same parameter, so collapsing them here means every consumer (the
    // value path, the pattern re-serializer, `parameter_names`) handles the
    // parenthesised spelling without a second arm. Spelling and the
    // name-only restriction are on `lex_parameter`.
    Parameter(String), // $param_name / $(param_name)

    LParen,      // (
    RParen,      // )
    LBracket,    // [
    RBracket,    // ]
    LBrace,      // {
    RBrace,      // }
    Colon,       // :
    Comma,       // ,
    Dot,         // .
    Semicolon,   // ;
    Dash,        // -
    GreaterThan, // >
    LessThan,    // <
    Star,        // *
    DotDot,      // ..

    Equals,            // =
    NotEquals,         // <>
    LessThanEquals,    // <=
    GreaterThanEquals, // >=

    RegexMatch, // =~

    Plus,       // +
    Slash,      // /
    Percent,    // %
    Pipe,       // |
    DoublePipe, // ||

    Identifier(String),
    StringLit(String),
    IntLit(i64),
    FloatLit(f64),

    /// `/*` — the opening of a block comment, which this dialect does not
    /// implement. Emitted as a marker rather than raised as a tokenizer
    /// error so `parse_cypher` can report it through the ordinary
    /// position machinery (tokenizer errors carry no line/col); the
    /// parser never sees it, because `parse_cypher` rejects the token
    /// stream first.
    BlockCommentOpen,
}

/// Tokenizer output: positioned tokens plus the verbatim source
/// lexeme of every keyword token.
///
/// Keyword tokens are unit variants — the tokenizer canonicalises
/// their case so the parser can match them structurally. Name-position
/// consumers (property keys, labels, rel types, aliases) must instead
/// see the **verbatim source spelling** (`{order: 1}` stores key
/// `order`, not `ORDER`), so the tokenizer records the original
/// lexeme for each keyword token here, keyed by token index.
pub struct TokenizedCypher {
    /// `(token, char-position of token start)` pairs.
    pub tokens: Vec<(CypherToken, usize)>,
    /// `(token index, verbatim source lexeme)` for every keyword
    /// token. Sparse — identifiers carry their own string.
    pub keyword_lexemes: Vec<(usize, String)>,
}

/// Position-stripping wrapper kept for the tokenizer's own tests
/// (which assert on `Vec<CypherToken>` directly). Production code
/// goes through [`tokenize_cypher_with_positions`] via
/// `parse_cypher`. 0.9.0 Cluster 3.
#[cfg(test)]
pub fn tokenize_cypher(input: &str) -> Result<Vec<CypherToken>, String> {
    Ok(tokenize_cypher_with_positions(input)?
        .tokens
        .into_iter()
        .map(|(tok, _pos)| tok)
        .collect())
}

/// Lex one parameter reference starting at the `$` in `chars[at]`.
///
/// Accepts both spellings — `$name` and the Neo4j 5 parenthesised `$(name)` —
/// and returns `(name, name_start, index_after)`. `$(...)` takes a parameter
/// *name*, not a general expression, and says so when handed one.
fn lex_parameter(chars: &[char], at: usize) -> Result<(String, usize, usize), String> {
    let len = chars.len();
    let mut i = at + 1; // consume $
    let parenthesised = i < len && chars[i] == '(';
    if parenthesised {
        i += 1; // consume (
    }
    let start = i;
    while i < len && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
        i += 1;
    }
    if i == start {
        return Err(format!(
            "Expected parameter name after '$' at position {}",
            start
        ));
    }
    let name: String = chars[start..i].iter().collect();
    if parenthesised {
        if i >= len || chars[i] != ')' {
            return Err(format!(
                "Expected ')' closing '$({}' at position {}. \
                 $(...) takes a parameter name only.",
                name, i
            ));
        }
        i += 1; // consume )
    }
    Ok((name, start, i))
}

/// Same as [`tokenize_cypher`] but returns the **char-position** at
/// the start of each token, alongside the token — plus the verbatim
/// keyword lexeme table (see [`TokenizedCypher`]). 0.9.0 Cluster 3 —
/// the parser uses positions to point at the exact token in error
/// messages instead of the prior approximate token re-walk.
///
/// Char-position is the index into `input.chars().collect()` —
/// `parse_cypher` walks the input to turn it into 1-based line/col on
/// the error path only, so nothing parallel is kept for the hot path.
pub fn tokenize_cypher_with_positions(input: &str) -> Result<TokenizedCypher, String> {
    let mut tokens: Vec<(CypherToken, usize)> = Vec::new();
    let mut keyword_lexemes: Vec<(usize, String)> = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];
        let start = i;

        if ch.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        if let Some(next) = take_comment(&chars, i, start, &mut tokens) {
            i = next;
            continue;
        }

        match ch {
            '(' => {
                tokens.push((CypherToken::LParen, start));
                i += 1;
            }
            ')' => {
                tokens.push((CypherToken::RParen, start));
                i += 1;
            }
            '[' => {
                tokens.push((CypherToken::LBracket, start));
                i += 1;
            }
            ']' => {
                tokens.push((CypherToken::RBracket, start));
                i += 1;
            }
            '{' => {
                tokens.push((CypherToken::LBrace, start));
                i += 1;
            }
            '}' => {
                tokens.push((CypherToken::RBrace, start));
                i += 1;
            }
            ':' => {
                tokens.push((CypherToken::Colon, start));
                i += 1;
            }
            ',' => {
                tokens.push((CypherToken::Comma, start));
                i += 1;
            }
            ';' => {
                tokens.push((CypherToken::Semicolon, start));
                i += 1;
            }
            '*' => {
                tokens.push((CypherToken::Star, start));
                i += 1;
            }
            '+' => {
                tokens.push((CypherToken::Plus, start));
                i += 1;
            }
            '/' => {
                tokens.push((CypherToken::Slash, start));
                i += 1;
            }
            '%' => {
                tokens.push((CypherToken::Percent, start));
                i += 1;
            }
            '|' => {
                if i + 1 < len && chars[i + 1] == '|' {
                    tokens.push((CypherToken::DoublePipe, start));
                    i += 2;
                } else {
                    tokens.push((CypherToken::Pipe, start));
                    i += 1;
                }
            }
            '=' => {
                if i + 1 < chars.len() && chars[i + 1] == '~' {
                    tokens.push((CypherToken::RegexMatch, start));
                    i += 2;
                } else {
                    tokens.push((CypherToken::Equals, start));
                    i += 1;
                }
            }

            '-' => {
                // Could be dash (edge syntax) or negative number in some contexts,
                // but we always tokenize as Dash and let the parser handle unary negation
                tokens.push((CypherToken::Dash, start));
                i += 1;
            }

            '<' => {
                if i + 1 < len && chars[i + 1] == '>' {
                    tokens.push((CypherToken::NotEquals, start));
                    i += 2;
                } else if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push((CypherToken::LessThanEquals, start));
                    i += 2;
                } else {
                    tokens.push((CypherToken::LessThan, start));
                    i += 1;
                }
            }

            '>' => {
                if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push((CypherToken::GreaterThanEquals, start));
                    i += 2;
                } else {
                    tokens.push((CypherToken::GreaterThan, start));
                    i += 1;
                }
            }

            '!' => {
                if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push((CypherToken::NotEquals, start));
                    i += 2;
                } else {
                    return Err(format!(
                        "Unexpected character '!' at position {}. Did you mean '!='?",
                        i
                    ));
                }
            }

            '.' => {
                if i + 1 < len && chars[i + 1] == '.' {
                    tokens.push((CypherToken::DotDot, start));
                    i += 2;
                } else if i + 1 < len && chars[i + 1].is_ascii_digit() {
                    // Float starting with dot: .5
                    let start = i;
                    i += 1;
                    while i < len && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                    let num_str: String = chars[start..i].iter().collect();
                    let f: f64 = num_str
                        .parse()
                        .map_err(|_| format!("Invalid float: {}", num_str))?;
                    tokens.push((CypherToken::FloatLit(f), start));
                } else {
                    tokens.push((CypherToken::Dot, start));
                    i += 1;
                }
            }

            '"' | '\'' => {
                let quote = ch;
                i += 1; // consume opening quote
                let mut s = String::new();
                let mut closed = false;
                while i < len {
                    if chars[i] == quote {
                        i += 1; // consume closing quote
                        closed = true;
                        break;
                    }
                    if chars[i] == '\\' && i + 1 < len {
                        i += 1;
                        match chars[i] {
                            'n' => {
                                s.push('\n');
                                i += 1;
                            }
                            't' => {
                                s.push('\t');
                                i += 1;
                            }
                            'r' => {
                                s.push('\r');
                                i += 1;
                            }
                            '\\' => {
                                s.push('\\');
                                i += 1;
                            }
                            // \uXXXX — 4-hex-digit unicode escape (openCypher /
                            // the form json.dumps emits, e.g. `—` → em-dash).
                            // Previously fell into the `other` arm below, which
                            // dropped the backslash and stored the literal text
                            // `u2014`. Decode when four valid hex digits follow;
                            // otherwise keep the literal `u` (lenient, no error).
                            'u' if i + 4 < len
                                && chars[i + 1..i + 5].iter().all(char::is_ascii_hexdigit) =>
                            {
                                let hex: String = chars[i + 1..i + 5].iter().collect();
                                match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                                    Some(decoded) => {
                                        s.push(decoded);
                                        i += 5;
                                    }
                                    // Valid hex but not a scalar value (surrogate)
                                    // — keep the literal `u`, leave digits in place.
                                    None => {
                                        s.push('u');
                                        i += 1;
                                    }
                                }
                            }
                            c if c == quote => {
                                s.push(c);
                                i += 1;
                            }
                            other => {
                                s.push(other);
                                i += 1;
                            }
                        }
                    } else {
                        s.push(chars[i]);
                        i += 1;
                    }
                }
                if !closed {
                    return Err(format!("Unterminated string literal: {}{}", quote, s));
                }
                tokens.push((CypherToken::StringLit(s), start));
            }

            c if c.is_ascii_digit() => {
                let start = i;
                let mut has_dot = false;
                while i < len && (chars[i].is_ascii_digit() || (chars[i] == '.' && !has_dot)) {
                    if chars[i] == '.' {
                        // Check for '..' (range operator) - don't consume
                        if i + 1 < len && chars[i + 1] == '.' {
                            break;
                        }
                        // Check if next char is a digit (decimal point) or not (property access after number)
                        if i + 1 >= len || !chars[i + 1].is_ascii_digit() {
                            break;
                        }
                        has_dot = true;
                    }
                    i += 1;
                }
                // Scientific notation: e.g. 1e6, 1.5e-3, 2E+10
                if i < len && (chars[i] == 'e' || chars[i] == 'E') {
                    has_dot = true; // Force float parsing
                    i += 1;
                    if i < len && (chars[i] == '+' || chars[i] == '-') {
                        i += 1;
                    }
                    while i < len && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let num_str: String = chars[start..i].iter().collect();
                if has_dot {
                    let f: f64 = num_str
                        .parse()
                        .map_err(|_| format!("Invalid float: {}", num_str))?;
                    tokens.push((CypherToken::FloatLit(f), start));
                } else {
                    match num_str.parse::<i64>() {
                        Ok(n) => tokens.push((CypherToken::IntLit(n), start)),
                        Err(_) => {
                            // i64::MIN is the only integer whose magnitude
                            // overflows i64::from_str (i64::MAX is 2^63-1,
                            // |i64::MIN| is 2^63). The unary-minus path is
                            // parsed as a Dash token followed by the
                            // positive literal — so `-9223372036854775808`
                            // is unrepresentable through the normal
                            // route, so the Dash is folded in here instead.
                            if num_str == "9223372036854775808"
                                && tokens
                                    .last()
                                    .is_some_and(|(t, _)| matches!(t, CypherToken::Dash))
                            {
                                let (_, dash_pos) = tokens.pop().unwrap();
                                tokens.push((CypherToken::IntLit(i64::MIN), dash_pos));
                            } else {
                                return Err(format!("Invalid integer: {}", num_str));
                            }
                        }
                    }
                }
            }

            '$' => {
                let (name, start, next) = lex_parameter(&chars, i)?;
                i = next;
                tokens.push((CypherToken::Parameter(name), start));
            }

            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < len && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                match keyword_token(&ident) {
                    Some(tok) => {
                        keyword_lexemes.push((tokens.len(), ident));
                        tokens.push((tok, start));
                    }
                    None => tokens.push((CypherToken::Identifier(ident), start)),
                }
            }

            // Backtick-quoted identifiers: `My Identifier`. The escaping rules,
            // and the injection the doubled backtick closes, are on
            // `scan_backtick_identifier`.
            '`' => {
                i += 1; // consume opening backtick
                let start = i;
                let (ident, next) = scan_backtick_identifier(&chars, i)?;
                i = next;
                tokens.push((CypherToken::Identifier(ident), start));
            }

            _ => {
                return Err(format!("Unexpected character '{}' at position {}", ch, i));
            }
        }
    }

    Ok(TokenizedCypher {
        tokens,
        keyword_lexemes,
    })
}

/// Scan a backtick-quoted identifier with `i` positioned just past the
/// opening backtick. A doubled backtick escapes to one literal backtick;
/// a single backtick terminates. Returns the identifier and the index just
/// past the closing backtick, or an unterminated-identifier error carrying
/// the partial text.
///
/// A doubled backtick is an escaped one, per openCypher: `` `a``b` `` is the
/// single identifier ``a`b``. Without the escape the quoted form had **no**
/// way to represent a backtick, and the tokenizer simply stopped at the first
/// one — so a caller that string-built a label or a variable from untrusted
/// input could close the quote and append clauses:
///
/// ```text
///   label = "Person`) DETACH DELETE n //"
///   MATCH (n:`Person`) DETACH DELETE n //`) RETURN count(n) AS c
/// ```
///
/// …which deleted every Person and reported a count. With doubling in place
/// the same input is representable as one (weird) identifier, so an emitter
/// has an escaping rule to apply and the break-out is closed at the grammar
/// rather than per binding.
fn scan_backtick_identifier(chars: &[char], mut i: usize) -> Result<(String, usize), String> {
    let len = chars.len();
    let mut ident = String::new();
    while i < len {
        if chars[i] == '`' {
            if i + 1 < len && chars[i + 1] == '`' {
                ident.push('`');
                i += 2;
                continue;
            }
            return Ok((ident, i + 1));
        }
        ident.push(chars[i]);
        i += 1;
    }
    Err(format!("Unterminated backtick identifier: `{}", ident))
}

/// Keyword token for an identifier lexeme, or `None` when the lexeme
/// is a plain identifier. Case-insensitive (Cypher keywords are).
fn keyword_token(ident: &str) -> Option<CypherToken> {
    let tok = match ident.to_uppercase().as_str() {
        "MATCH" => CypherToken::Match,
        "OPTIONAL" => CypherToken::Optional,
        "WHERE" => CypherToken::Where,
        "RETURN" => CypherToken::Return,
        "WITH" => CypherToken::With,
        "ORDER" => CypherToken::Order,
        "BY" => CypherToken::By,
        "AS" => CypherToken::As,
        "AND" => CypherToken::And,
        "OR" => CypherToken::Or,
        "NOT" => CypherToken::Not,
        "IN" => CypherToken::In,
        "IS" => CypherToken::Is,
        "NULL" => CypherToken::Null,
        "NULLS" => CypherToken::Nulls,
        "LIMIT" => CypherToken::Limit,
        "SKIP" => CypherToken::Skip,
        "UNWIND" => CypherToken::Unwind,
        "UNION" => CypherToken::Union,
        "INTERSECT" => CypherToken::Intersect,
        "EXCEPT" => CypherToken::Except,
        "ALL" => CypherToken::All,
        "DISTINCT" => CypherToken::Distinct,
        "CREATE" => CypherToken::Create,
        "SET" => CypherToken::Set,
        "DELETE" => CypherToken::Delete,
        "DETACH" => CypherToken::Detach,
        "MERGE" => CypherToken::Merge,
        "REMOVE" => CypherToken::Remove,
        "FOREACH" => CypherToken::Foreach,
        "ON" => CypherToken::On,
        "ASC" | "ASCENDING" => CypherToken::Asc,
        "DESC" | "DESCENDING" => CypherToken::Desc,
        "CASE" => CypherToken::Case,
        "WHEN" => CypherToken::When,
        "THEN" => CypherToken::Then,
        "ELSE" => CypherToken::Else,
        "END" => CypherToken::End,
        "TRUE" => CypherToken::True,
        "FALSE" => CypherToken::False,
        "STARTS" => CypherToken::StartsWith,
        "ENDS" => CypherToken::EndsWith,
        "CONTAINS" => CypherToken::Contains,
        "EXISTS" => CypherToken::Exists,
        "EXPLAIN" => CypherToken::Explain,
        "PROFILE" => CypherToken::Profile,
        "CALL" => CypherToken::Call,
        "YIELD" => CypherToken::Yield,
        "OVER" => CypherToken::Over,
        "PARTITION" => CypherToken::Partition,
        "HAVING" => CypherToken::Having,
        "XOR" => CypherToken::Xor,
        _ => return None,
    };
    Some(tok)
}

/// Lowercase source word for a keyword token — the spelling used for `AS`
/// aliases and, uppercased, for keywords in parser error messages. `None`
/// for symbols, literals and identifiers.
pub fn token_to_keyword_name(token: &CypherToken) -> Option<String> {
    let name = match token {
        CypherToken::Match => "match",
        CypherToken::Optional => "optional",
        CypherToken::Where => "where",
        CypherToken::Return => "return",
        CypherToken::With => "with",
        CypherToken::Order => "order",
        CypherToken::By => "by",
        CypherToken::As => "as",
        CypherToken::And => "and",
        CypherToken::Or => "or",
        CypherToken::Not => "not",
        CypherToken::In => "in",
        CypherToken::Is => "is",
        CypherToken::Null => "null",
        CypherToken::Nulls => "nulls",
        CypherToken::Limit => "limit",
        CypherToken::Skip => "skip",
        CypherToken::Unwind => "unwind",
        CypherToken::Union => "union",
        CypherToken::Intersect => "intersect",
        CypherToken::Except => "except",
        CypherToken::All => "all",
        CypherToken::Distinct => "distinct",
        CypherToken::Create => "create",
        CypherToken::Set => "set",
        CypherToken::Delete => "delete",
        CypherToken::Detach => "detach",
        CypherToken::Merge => "merge",
        CypherToken::Remove => "remove",
        CypherToken::Foreach => "foreach",
        CypherToken::On => "on",
        CypherToken::Asc => "asc",
        CypherToken::Desc => "desc",
        CypherToken::StartsWith => "starts",
        CypherToken::EndsWith => "ends",
        CypherToken::Contains => "contains",
        CypherToken::Case => "case",
        CypherToken::When => "when",
        CypherToken::Then => "then",
        CypherToken::Else => "else",
        CypherToken::End => "end",
        CypherToken::True => "true",
        CypherToken::False => "false",
        CypherToken::Exists => "exists",
        CypherToken::Explain => "explain",
        CypherToken::Profile => "profile",
        CypherToken::Call => "call",
        CypherToken::Yield => "yield",
        CypherToken::Over => "over",
        CypherToken::Partition => "partition",
        CypherToken::Having => "having",
        CypherToken::Xor => "xor",
        _ => return None,
    };
    Some(name.to_string())
}

/// Canonical UPPERCASE word for a keyword token used as a NAME (relationship
/// type, node label, or property key) — KG-2 soft keywords. Returns `None` for
/// non-keyword tokens AND for keywords that must stay reserved even in name
/// position.
///
/// This decides *whether* a keyword is soft-reservable; the actual NAME the
/// parser stores is the **verbatim source lexeme** from
/// [`TokenizedCypher::keyword_lexemes`] (`{order: 1}` → key `order`, matching
/// Neo4j's case-preserving property keys). The canonical uppercase returned
/// here is only the fallback for a token index missing from that table.
///
/// Distinct from `token_to_keyword_name` (lowercase, for `AS` aliases).
///
/// The SAFE set is the operator / comparison / sort / set / mutation keywords —
/// words that, inside a pattern, can only be a name (they appear elsewhere only
/// in WHERE-expression or clause position, which the re-serializer reaches at
/// bracket/paren depth 0, before this is ever consulted). Deliberately kept
/// reserved (→ `None`): the clause-flow words (MATCH / OPTIONAL / WHERE /
/// RETURN / WITH / UNWIND / LIMIT / SKIP, AND / OR), the value literals
/// (NULL / NULLS / TRUE / FALSE), and the value-expression words (CASE / WHEN /
/// THEN / ELSE / END, EXISTS) — because those can legitimately appear as a
/// property *value* in an inline map (`{x: null}`) and must not be mis-read as
/// a name. The backtick escape hatch still works for any excluded word.
///
/// The three *value-literal* words it excludes — TRUE / FALSE / NULL — are
/// handled by [`reserved_literal_name_token`] instead, because for them the
/// grammar position, not a table, decides.
pub fn keyword_name_token(token: &CypherToken) -> Option<&'static str> {
    let name = match token {
        CypherToken::Contains => "CONTAINS",
        CypherToken::StartsWith => "STARTS",
        CypherToken::EndsWith => "ENDS",
        CypherToken::In => "IN",
        CypherToken::Is => "IS",
        CypherToken::Not => "NOT",
        CypherToken::Xor => "XOR",
        CypherToken::Order => "ORDER",
        CypherToken::By => "BY",
        CypherToken::Asc => "ASC",
        CypherToken::Desc => "DESC",
        CypherToken::Distinct => "DISTINCT",
        CypherToken::All => "ALL",
        CypherToken::On => "ON",
        CypherToken::Over => "OVER",
        CypherToken::Partition => "PARTITION",
        CypherToken::Having => "HAVING",
        CypherToken::Detach => "DETACH",
        CypherToken::Merge => "MERGE",
        CypherToken::Create => "CREATE",
        CypherToken::Delete => "DELETE",
        CypherToken::Set => "SET",
        CypherToken::Remove => "REMOVE",
        CypherToken::Foreach => "FOREACH",
        CypherToken::Yield => "YIELD",
        CypherToken::Call => "CALL",
        CypherToken::Union => "UNION",
        CypherToken::Intersect => "INTERSECT",
        CypherToken::Except => "EXCEPT",
        CypherToken::Explain => "EXPLAIN",
        CypherToken::Profile => "PROFILE",
        CypherToken::As => "AS",
        _ => return None,
    };
    Some(name)
}

/// Handle a comment opener at `i`, reporting the index to resume from, or
/// `None` when `i` is not a comment opener.
///
/// A `//` line comment is skipped to end of line. A `/*` block comment is
/// **not** this dialect's syntax, and is marked with
/// [`CypherToken::BlockCommentOpen`] rather than failing here — see that
/// variant for why the tokenizer does not raise the error itself.
fn take_comment(
    chars: &[char],
    i: usize,
    start: usize,
    tokens: &mut Vec<(CypherToken, usize)>,
) -> Option<usize> {
    if chars[i] != '/' {
        return None;
    }
    match chars.get(i + 1) {
        Some('/') => {
            let mut end = i;
            while end < chars.len() && chars[end] != '\n' {
                end += 1;
            }
            Some(end)
        }
        Some('*') => {
            tokens.push((CypherToken::BlockCommentOpen, start));
            Some(i + 2)
        }
        _ => None,
    }
}

/// Render a token the way the user wrote it, for error messages.
///
/// Parser errors used to interpolate `{:?}`, which leaked Rust shapes into
/// user-facing output — `Expected variable name in SET, got Some(IntLit(1))`
/// for `SET 1 = 2`. Literals and identifiers render as their source text,
/// symbols as the character they are, and keywords as the canonical
/// uppercase word.
pub fn describe_token(token: &CypherToken) -> String {
    match token {
        CypherToken::Identifier(name) => name.clone(),
        CypherToken::StringLit(s) => format!("'{s}'"),
        CypherToken::IntLit(n) => n.to_string(),
        CypherToken::FloatLit(f) => f.to_string(),
        CypherToken::Parameter(name) => format!("${name}"),
        other => match token_symbol(other) {
            Some(symbol) => symbol.to_string(),
            None => token_to_keyword_name(other)
                .map(|word| word.to_uppercase())
                .unwrap_or_else(|| format!("{other:?}")),
        },
    }
}

/// [`describe_token`] for a lookahead that may have run off the end.
pub fn describe_token_opt(token: Option<&CypherToken>) -> String {
    match token {
        Some(token) => describe_token(token),
        None => "end of input".to_string(),
    }
}

/// The source spelling of a punctuation/operator token. `None` for
/// keywords, literals and identifiers, which [`describe_token`] renders
/// from their own payload.
fn token_symbol(token: &CypherToken) -> Option<&'static str> {
    Some(match token {
        CypherToken::LParen => "(",
        CypherToken::RParen => ")",
        CypherToken::LBracket => "[",
        CypherToken::RBracket => "]",
        CypherToken::LBrace => "{",
        CypherToken::RBrace => "}",
        CypherToken::Colon => ":",
        CypherToken::Comma => ",",
        CypherToken::Dot => ".",
        CypherToken::Semicolon => ";",
        CypherToken::Dash => "-",
        CypherToken::GreaterThan => ">",
        CypherToken::LessThan => "<",
        CypherToken::Star => "*",
        CypherToken::DotDot => "..",
        CypherToken::Equals => "=",
        CypherToken::NotEquals => "<>",
        CypherToken::LessThanEquals => "<=",
        CypherToken::GreaterThanEquals => ">=",
        CypherToken::RegexMatch => "=~",
        CypherToken::Plus => "+",
        CypherToken::Slash => "/",
        CypherToken::Percent => "%",
        CypherToken::Pipe => "|",
        CypherToken::DoublePipe => "||",
        CypherToken::BlockCommentOpen => "/*",
        _ => return None,
    })
}

/// Canonical UPPERCASE word for one of the three **value-literal** keywords —
/// TRUE, FALSE, NULL — when it is used as a NAME. Returns `None` for every
/// other token.
///
/// These are deliberately NOT in [`keyword_name_token`]. That table answers
/// "may this word be a name *anywhere* a name is expected", and its members are
/// safe there because they cannot also be a value. TRUE / FALSE / NULL can be
/// both, so only the grammar position separates them: openCypher 9 spells a
/// label, relationship type or property key as
/// `SchemaName = SymbolicName | ReservedWord`, and lists all three under
/// `ReservedWord` — while a variable is `SymbolicName` alone, which excludes
/// them. Neo4j 25 matches that for the schema half
/// (`labelType : COLON symbolicNameString`, whose unescaped alternatives
/// include TRUE / FALSE / NULL).
///
/// So the consumers are the *name* positions only:
/// [`super::parser::CypherParser::expect_name`] (label, relationship type,
/// property key, map key) and the MATCH-pattern re-serializer's name arm. A
/// value position never consults this and keeps reading the literal
/// (`{x: true}`, `WHERE n.x = true`, `RETURN null`). As with the soft keywords
/// the stored NAME is the verbatim lexeme; this word is only the fallback for
/// a token index missing from that table.
pub fn reserved_literal_name_token(token: &CypherToken) -> Option<&'static str> {
    match token {
        CypherToken::True => Some("TRUE"),
        CypherToken::False => Some("FALSE"),
        CypherToken::Null => Some("NULL"),
        _ => None,
    }
}

// Literal test inputs like `3.14` trip approx_constant.
#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_match_return() {
        let tokens = tokenize_cypher("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(
            tokens,
            vec![
                CypherToken::Match,
                CypherToken::LParen,
                CypherToken::Identifier("n".to_string()),
                CypherToken::Colon,
                CypherToken::Identifier("Person".to_string()),
                CypherToken::RParen,
                CypherToken::Return,
                CypherToken::Identifier("n".to_string()),
            ]
        );
    }

    #[test]
    fn test_where_with_comparison() {
        let tokens = tokenize_cypher("WHERE n.age > 30 AND n.name = 'Alice'").unwrap();
        assert_eq!(
            tokens,
            vec![
                CypherToken::Where,
                CypherToken::Identifier("n".to_string()),
                CypherToken::Dot,
                CypherToken::Identifier("age".to_string()),
                CypherToken::GreaterThan,
                CypherToken::IntLit(30),
                CypherToken::And,
                CypherToken::Identifier("n".to_string()),
                CypherToken::Dot,
                CypherToken::Identifier("name".to_string()),
                CypherToken::Equals,
                CypherToken::StringLit("Alice".to_string()),
            ]
        );
    }

    #[test]
    fn test_not_equals() {
        let tokens = tokenize_cypher("n.x <> 5").unwrap();
        assert!(tokens.contains(&CypherToken::NotEquals));
    }

    #[test]
    fn test_less_than_equals() {
        let tokens = tokenize_cypher("n.x <= 10").unwrap();
        assert!(tokens.contains(&CypherToken::LessThanEquals));
    }

    #[test]
    fn test_greater_than_equals() {
        let tokens = tokenize_cypher("n.x >= 10").unwrap();
        assert!(tokens.contains(&CypherToken::GreaterThanEquals));
    }

    #[test]
    fn test_return_with_alias() {
        let tokens = tokenize_cypher("RETURN n.name AS name, count(n) AS total").unwrap();
        assert!(tokens.contains(&CypherToken::As));
        assert!(tokens.contains(&CypherToken::Return));
    }

    #[test]
    fn test_order_by_limit() {
        let tokens = tokenize_cypher("ORDER BY n.age DESC LIMIT 10").unwrap();
        assert!(tokens.contains(&CypherToken::Order));
        assert!(tokens.contains(&CypherToken::By));
        assert!(tokens.contains(&CypherToken::Desc));
        assert!(tokens.contains(&CypherToken::Limit));
    }

    #[test]
    fn test_string_escapes() {
        let tokens = tokenize_cypher(r#"'it\'s a \"test\"'"#).unwrap();
        if let CypherToken::StringLit(s) = &tokens[0] {
            assert_eq!(s, "it's a \"test\"");
        } else {
            panic!("Expected string literal");
        }
    }

    #[test]
    fn test_string_unicode_escape() {
        // \uXXXX must decode, not drop the backslash (petekSuite bug 3:
        // `—` was stored as the literal text `u2014`). The raw-string
        // input below contains a real backslash-u-2014 escape sequence.
        let tokens = tokenize_cypher(r#""A\u2014B""#).unwrap();
        assert_eq!(tokens[0], CypherToken::StringLit("A\u{2014}B".to_string()));
        // A non-\uXXXX backslash-u stays lenient (literal `u`).
        let tokens = tokenize_cypher(r#""x\uZZZZy""#).unwrap();
        assert_eq!(tokens[0], CypherToken::StringLit("xuZZZZy".to_string()));
    }

    #[test]
    fn test_float_literal() {
        let tokens = tokenize_cypher("3.14").unwrap();
        assert_eq!(tokens, vec![CypherToken::FloatLit(3.14)]);
    }

    #[test]
    fn test_case_insensitive_keywords() {
        let tokens = tokenize_cypher("match (n) where n.x = 1 return n").unwrap();
        assert_eq!(tokens[0], CypherToken::Match);
        assert_eq!(tokens[4], CypherToken::Where);
        assert_eq!(tokens[10], CypherToken::Return);
    }

    #[test]
    fn test_edge_pattern_tokens() {
        let tokens = tokenize_cypher("(a)-[:KNOWS]->(b)").unwrap();
        assert_eq!(
            tokens,
            vec![
                CypherToken::LParen,
                CypherToken::Identifier("a".to_string()),
                CypherToken::RParen,
                CypherToken::Dash,
                CypherToken::LBracket,
                CypherToken::Colon,
                CypherToken::Identifier("KNOWS".to_string()),
                CypherToken::RBracket,
                CypherToken::Dash,
                CypherToken::GreaterThan,
                CypherToken::LParen,
                CypherToken::Identifier("b".to_string()),
                CypherToken::RParen,
            ]
        );
    }

    #[test]
    fn test_null_checks() {
        let tokens = tokenize_cypher("WHERE n.x IS NULL").unwrap();
        assert!(tokens.contains(&CypherToken::Is));
        assert!(tokens.contains(&CypherToken::Null));
    }

    #[test]
    fn test_not_null() {
        let tokens = tokenize_cypher("WHERE n.x IS NOT NULL").unwrap();
        assert!(tokens.contains(&CypherToken::Is));
        assert!(tokens.contains(&CypherToken::Not));
        assert!(tokens.contains(&CypherToken::Null));
    }

    #[test]
    fn test_backtick_identifier() {
        let tokens = tokenize_cypher("`My Node`").unwrap();
        assert_eq!(tokens, vec![CypherToken::Identifier("My Node".to_string())]);
    }

    #[test]
    fn test_in_list() {
        let tokens = tokenize_cypher("WHERE n.x IN [1, 2, 3]").unwrap();
        assert!(tokens.contains(&CypherToken::In));
        assert!(tokens.contains(&CypherToken::LBracket));
        assert!(tokens.contains(&CypherToken::RBracket));
    }

    #[test]
    fn test_var_length_path() {
        let tokens = tokenize_cypher("-[:KNOWS*1..3]->").unwrap();
        assert!(tokens.contains(&CypherToken::Star));
        assert!(tokens.contains(&CypherToken::DotDot));
    }

    #[test]
    fn test_case_tokens() {
        let tokens = tokenize_cypher("CASE WHEN x THEN 1 ELSE 0 END").unwrap();
        assert_eq!(tokens[0], CypherToken::Case);
        assert_eq!(tokens[1], CypherToken::When);
        assert_eq!(tokens[3], CypherToken::Then);
        assert_eq!(tokens[5], CypherToken::Else);
        assert_eq!(tokens[7], CypherToken::End);
    }

    #[test]
    fn test_case_insensitive_case() {
        let tokens = tokenize_cypher("case when x then 1 else 0 end").unwrap();
        assert_eq!(tokens[0], CypherToken::Case);
        assert_eq!(tokens[1], CypherToken::When);
    }

    #[test]
    fn test_parameter_token() {
        let tokens = tokenize_cypher("$min_age").unwrap();
        assert_eq!(tokens, vec![CypherToken::Parameter("min_age".to_string())]);
    }

    #[test]
    fn test_parameter_in_query() {
        let tokens = tokenize_cypher("WHERE n.age > $age AND n.city = $city").unwrap();
        assert!(tokens.contains(&CypherToken::Parameter("age".to_string())));
        assert!(tokens.contains(&CypherToken::Parameter("city".to_string())));
    }

    #[test]
    fn parenthesised_parameter_is_the_same_token_as_the_bare_form() {
        assert_eq!(
            tokenize_cypher("MATCH (n:$(label))").unwrap(),
            tokenize_cypher("MATCH (n:$label)").unwrap()
        );
        assert_eq!(
            tokenize_cypher("$(min_age)").unwrap(),
            vec![CypherToken::Parameter("min_age".to_string())]
        );
    }

    #[test]
    fn unclosed_or_empty_parenthesised_parameter_is_a_syntax_error() {
        assert!(tokenize_cypher("MATCH (n:$(label").is_err());
        assert!(tokenize_cypher("MATCH (n:$())").is_err());
        // `$(...)` takes a name, not an expression.
        assert!(tokenize_cypher("MATCH (n:$(row.label))").is_err());
    }

    #[test]
    fn test_parameter_empty_name_error() {
        let result = tokenize_cypher("$");
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_remove_on_tokens() {
        let tokens = tokenize_cypher("MERGE REMOVE ON").unwrap();
        assert_eq!(tokens[0], CypherToken::Merge);
        assert_eq!(tokens[1], CypherToken::Remove);
        assert_eq!(tokens[2], CypherToken::On);
    }
}
