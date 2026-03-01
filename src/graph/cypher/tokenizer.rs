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
    Limit,
    Skip,
    Unwind,
    Union,
    All,
    Distinct,
    Create,
    Set,
    Delete,
    Detach,
    Merge,
    Remove,
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

pub fn tokenize_cypher(input: &str) -> Result<Vec<CypherToken>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];

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
                tokens.push(CypherToken::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(CypherToken::RParen);
                i += 1;
            }
            '[' => {
                tokens.push(CypherToken::LBracket);
                i += 1;
            }
            ']' => {
                tokens.push(CypherToken::RBracket);
                i += 1;
            }
            '{' => {
                tokens.push(CypherToken::LBrace);
                i += 1;
            }
            '}' => {
                tokens.push(CypherToken::RBrace);
                i += 1;
            }
            ':' => {
                tokens.push(CypherToken::Colon);
                i += 1;
            }
            ',' => {
                tokens.push(CypherToken::Comma);
                i += 1;
            }
            ';' => {
                tokens.push(CypherToken::Semicolon);
                i += 1;
            }
            '*' => {
                tokens.push(CypherToken::Star);
                i += 1;
            }
            '+' => {
                tokens.push(CypherToken::Plus);
                i += 1;
            }
            '/' => {
                tokens.push(CypherToken::Slash);
                i += 1;
            }
            '|' => {
                if i + 1 < len && chars[i + 1] == '|' {
                    tokens.push(CypherToken::DoublePipe);
                    i += 2;
                } else {
                    tokens.push(CypherToken::Pipe);
                    i += 1;
                }
            }
            '=' => {
                if i + 1 < chars.len() && chars[i + 1] == '~' {
                    tokens.push(CypherToken::RegexMatch);
                    i += 2;
                } else {
                    tokens.push(CypherToken::Equals);
                    i += 1;
                }
            }

            '-' => {
                // Could be dash (edge syntax) or negative number in some contexts,
                // but we always tokenize as Dash and let the parser handle unary negation
                tokens.push(CypherToken::Dash);
                i += 1;
            }

            '<' => {
                if i + 1 < len && chars[i + 1] == '>' {
                    tokens.push(CypherToken::NotEquals);
                    i += 2;
                } else if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push(CypherToken::LessThanEquals);
                    i += 2;
                } else {
                    tokens.push(CypherToken::LessThan);
                    i += 1;
                }
            }

            '>' => {
                if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push(CypherToken::GreaterThanEquals);
                    i += 2;
                } else {
                    tokens.push(CypherToken::GreaterThan);
                    i += 1;
                }
            }

            '!' => {
                if i + 1 < len && chars[i + 1] == '=' {
                    tokens.push(CypherToken::NotEquals);
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
                    tokens.push(CypherToken::DotDot);
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
                    tokens.push(CypherToken::FloatLit(f));
                } else {
                    tokens.push(CypherToken::Dot);
                    i += 1;
                }
            }

            // String literals
            '"' | '\'' => {
                let quote = ch;
                i += 1; // consume opening quote
                let mut s = String::new();
                while i < len {
                    if chars[i] == quote {
                        i += 1; // consume closing quote
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
                tokens.push(CypherToken::StringLit(s));
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
                    tokens.push(CypherToken::FloatLit(f));
                } else {
                    let n: i64 = num_str
                        .parse()
                        .map_err(|_| format!("Invalid integer: {}", num_str))?;
                    tokens.push(CypherToken::IntLit(n));
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
                tokens.push(CypherToken::Parameter(name));
            }

            // Identifiers and keywords
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < len && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                tokens.push(identifier_to_token(ident));
            }

            // Backtick-quoted identifiers: `My Identifier`
            '`' => {
                i += 1; // consume opening backtick
                let start = i;
                while i < len && chars[i] != '`' {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                if i < len {
                    i += 1; // consume closing backtick
                }
                tokens.push(CypherToken::Identifier(ident));
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
        "LIMIT" => CypherToken::Limit,
        "SKIP" => CypherToken::Skip,
        "UNWIND" => CypherToken::Unwind,
        "UNION" => CypherToken::Union,
        "ALL" => CypherToken::All,
        "DISTINCT" => CypherToken::Distinct,
        "CREATE" => CypherToken::Create,
        "SET" => CypherToken::Set,
        "DELETE" => CypherToken::Delete,
        "DETACH" => CypherToken::Detach,
        "MERGE" => CypherToken::Merge,
        "REMOVE" => CypherToken::Remove,
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
        CypherToken::Limit => "limit",
        CypherToken::Skip => "skip",
        CypherToken::Unwind => "unwind",
        CypherToken::Union => "union",
        CypherToken::All => "all",
        CypherToken::Distinct => "distinct",
        CypherToken::Create => "create",
        CypherToken::Set => "set",
        CypherToken::Delete => "delete",
        CypherToken::Detach => "detach",
        CypherToken::Merge => "merge",
        CypherToken::Remove => "remove",
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
        _ => return None,
    };
    Some(name.to_string())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
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
