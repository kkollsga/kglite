// src/graph/cypher/tokenizer.rs
// Cypher-level tokenizer handling keywords, operators, dot notation, and comparisons

// ============================================================================
// Token Types
// ============================================================================

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

    // Parameters
    Parameter(String), // $param_name

    // Symbols
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

    // Comparison operators
    Equals,            // =
    NotEquals,         // <>
    LessThanEquals,    // <=
    GreaterThanEquals, // >=

    // Regex
    RegexMatch, // =~

    // Arithmetic
    Plus,       // +
    Slash,      // /
    Percent,    // %
    Pipe,       // |
    DoublePipe, // ||

    // Literals and identifiers
    Identifier(String),
    StringLit(String),
    IntLit(i64),
    FloatLit(f64),
}

// ============================================================================
// Tokenizer
// ============================================================================

/// Position-stripping wrapper kept for the tokenizer's own tests
/// (which assert on `Vec<CypherToken>` directly). Production code
/// goes through [`tokenize_cypher_with_positions`] via
/// `parse_cypher`. 0.9.0 Cluster 3.
#[cfg(test)]
pub fn tokenize_cypher(input: &str) -> Result<Vec<CypherToken>, String> {
    Ok(tokenize_cypher_with_positions(input)?
        .into_iter()
        .map(|(tok, _pos)| tok)
        .collect())
}

/// Same as [`tokenize_cypher`] but returns the **char-position** at
/// the start of each token, alongside the token. 0.9.0 Cluster 3 — the
/// parser uses this to format byte-precise `(line, col)` in error
/// messages instead of the prior approximate token re-walk.
///
/// Char-position is the index into `input.chars().collect()` —
/// converted to byte offset / line:col by the consumer on error
/// (rare path; not worth a parallel byte-offset table for the hot
/// path).
pub fn tokenize_cypher_with_positions(input: &str) -> Result<Vec<(CypherToken, usize)>, String> {
    let mut tokens: Vec<(CypherToken, usize)> = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];
        // Position at the start of this token. Captured once per
        // loop iteration; tokens.push(...) callers below pair their
        // CypherToken with `start`. (0.9.0 Cluster 3.)
        let start = i;

        // Skip whitespace
        if ch.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Single-line comments: // to end of line
        if ch == '/' && i + 1 < len && chars[i + 1] == '/' {
            while i < len && chars[i] != '\n' {
                i += 1;
            }
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
                    i += 1; // skip the dot
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

            // String literals
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
                        s.push(match chars[i] {
                            'n' => '\n',
                            't' => '\t',
                            'r' => '\r',
                            '\\' => '\\',
                            c if c == quote => c,
                            other => other,
                        });
                        i += 1;
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

            // Numbers
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
                            // route. Look back: if we're directly after a
                            // Dash and the digit string is exactly 2^63,
                            // consume the Dash and emit IntLit(i64::MIN).
                            // Otherwise the literal is genuinely too large.
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

            // Parameter: $name
            '$' => {
                i += 1; // consume $
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
                tokens.push((CypherToken::Parameter(name), start));
            }

            // Identifiers and keywords
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < len && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                tokens.push((identifier_to_token(ident), start));
            }

            // Backtick-quoted identifiers: `My Identifier`
            '`' => {
                i += 1; // consume opening backtick
                let start = i;
                while i < len && chars[i] != '`' {
                    i += 1;
                }
                if i >= len {
                    let ident: String = chars[start..i].iter().collect();
                    return Err(format!("Unterminated backtick identifier: `{}", ident));
                }
                let ident: String = chars[start..i].iter().collect();
                i += 1; // consume closing backtick
                tokens.push((CypherToken::Identifier(ident), start));
            }

            _ => {
                return Err(format!("Unexpected character '{}' at position {}", ch, i));
            }
        }
    }

    Ok(tokens)
}

/// Convert an identifier string to the appropriate token (keyword or identifier)
fn identifier_to_token(ident: String) -> CypherToken {
    match ident.to_uppercase().as_str() {
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
        _ => CypherToken::Identifier(ident),
    }
}

/// Convert a keyword token back to its string form for use as an alias name.
/// Returns None for non-keyword tokens (symbols, literals, etc.).
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
/// Distinct from `token_to_keyword_name` (lowercase, for `AS` aliases): names
/// are case-sensitive and must round-trip verbatim (`[:CONTAINS]` stays
/// `CONTAINS`, not `contains`).
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

// ============================================================================
// Tests
// ============================================================================

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
