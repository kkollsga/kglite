// Parser — tokenizes and parses Cypher-like pattern strings into a Pattern AST.

use crate::datatypes::values::Value;
use std::collections::HashMap;
use std::iter::Peekable;
use std::str::Chars;

use super::pattern::{
    EdgeDirection, EdgePattern, NodePattern, ParamLabel, Pattern, PatternElement, PropertyMatcher,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    LParen,      // (
    RParen,      // )
    LBracket,    // [
    RBracket,    // ]
    LBrace,      // {
    RBrace,      // }
    Colon,       // :
    Comma,       // ,
    Dash,        // -
    GreaterThan, // >
    LessThan,    // <
    Star,        // * (for variable-length paths)
    DotDot,      // .. (for range in variable-length)
    Dot,         // . (property access in an inline-map value: {id: prior.id})
    Pipe,        // | (for multi-type edges: [:A|B])
    Identifier(String),
    StringLit(String),
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    Parameter(String), // $param_name
}

/// Lex one numeric literal, with the sign (if any) already consumed by the
/// caller and reported through `negative`.
///
/// Accepts `12`, `1.5` and the leading-dot form `.5` (normalised to `0.5`),
/// and stops before a `..` range operator so `*1..3` still lexes as
/// `IntLit(1) DotDot IntLit(3)`.
///
/// The sign is folded into the string that is parsed, never applied
/// afterwards: `-9223372036854775808` is `i64::MIN`, but its magnitude alone
/// does not fit in an `i64`, so a parse-then-negate lexer would reject it.
fn lex_number(chars: &mut Peekable<Chars<'_>>, negative: bool) -> Result<Token, String> {
    let mut num_str = String::new();
    if negative {
        num_str.push('-');
    }
    let mut has_dot = false;
    if chars.peek() == Some(&'.') {
        chars.next();
        num_str.push_str("0.");
        has_dot = true;
    }
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            num_str.push(c);
            chars.next();
        } else if c == '.' && !has_dot {
            // A `..` range operator ends the literal; a lone `.` is a decimal
            // point. Peek through a clone so neither is consumed here.
            let mut peek_chars = chars.clone();
            peek_chars.next();
            if peek_chars.peek() == Some(&'.') {
                break;
            }
            has_dot = true;
            num_str.push(c);
            chars.next();
        } else {
            break;
        }
    }
    if has_dot {
        Ok(Token::FloatLit(
            num_str
                .parse()
                .map_err(|_| format!("Invalid float: {}", num_str))?,
        ))
    } else {
        Ok(Token::IntLit(
            num_str
                .parse()
                .map_err(|_| format!("Invalid integer: {}", num_str))?,
        ))
    }
}

/// Does a `-` at the current position open a signed numeric literal?
/// True only when a digit, or a `.` followed by a digit, comes next —
/// structural dashes in a pattern are always followed by `[`, `>`, `(`,
/// `<` or another `-`, so an edge is never mistaken for a number.
fn opens_signed_number(chars: &Peekable<Chars<'_>>) -> bool {
    let mut ahead = chars.clone();
    match ahead.next() {
        Some(c) if c.is_ascii_digit() => true,
        Some('.') => ahead.next().is_some_and(|c| c.is_ascii_digit()),
        _ => false,
    }
}

/// Would `word`, written bare into a pattern string, lex as something other
/// than an [`Token::Identifier`]?
///
/// **This is the emitter's obligation, and it belongs here** — next to the
/// lexer that creates the hazard. Pattern strings are not written by users;
/// they are *re-serialized* from an already-tokenized Cypher query by
/// `languages::cypher::parser::match_pattern`, which has to reproduce every
/// name it received. An identifier that this tokenizer would read back as a
/// literal has to be emitted backtick-quoted, or the name silently changes
/// meaning in transit — which is exactly how a backticked `` `TRUE` `` label
/// could be created and never matched: the escape was dropped and the
/// secondary lexer re-read a boolean.
///
/// Keep this in step with the identifier arm of [`tokenize`]; the agreement is
/// pinned by `quoting_predicate_agrees_with_the_tokenizer`.
pub fn bare_word_needs_quoting(word: &str) -> bool {
    // A leading `$` lexes as a parameter reference here, which is how a
    // *dynamic* label is written (`(n:$label)`). A name that happens to start
    // with `$` — only reachable as `` `$label` `` in the source, since the
    // primary tokenizer would otherwise have made it a parameter — must
    // therefore be re-emitted quoted, or the re-serializer turns a literal
    // label into a parameter reference and the query silently changes meaning.
    word.starts_with('$') || word.eq_ignore_ascii_case("true") || word.eq_ignore_ascii_case("false")
}

pub fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            ' ' | '\t' | '\n' | '\r' => {
                chars.next();
            }
            '(' => {
                tokens.push(Token::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(Token::RParen);
                chars.next();
            }
            '[' => {
                tokens.push(Token::LBracket);
                chars.next();
            }
            ']' => {
                tokens.push(Token::RBracket);
                chars.next();
            }
            '{' => {
                tokens.push(Token::LBrace);
                chars.next();
            }
            '}' => {
                tokens.push(Token::RBrace);
                chars.next();
            }
            ':' => {
                tokens.push(Token::Colon);
                chars.next();
            }
            ',' => {
                tokens.push(Token::Comma);
                chars.next();
            }
            '-' => {
                chars.next();
                if opens_signed_number(&chars) {
                    // Negative inline-map literal, e.g. `{temp: -1}`. The
                    // sign belongs to the number, not to a structural dash.
                    tokens.push(lex_number(&mut chars, true)?);
                } else {
                    tokens.push(Token::Dash);
                }
            }
            '>' => {
                tokens.push(Token::GreaterThan);
                chars.next();
            }
            '<' => {
                tokens.push(Token::LessThan);
                chars.next();
            }
            '*' => {
                tokens.push(Token::Star);
                chars.next();
            }
            '|' => {
                tokens.push(Token::Pipe);
                chars.next();
            }
            '.' => {
                let mut ahead = chars.clone();
                ahead.next();
                if ahead.peek() == Some(&'.') {
                    chars.next();
                    chars.next();
                    tokens.push(Token::DotDot);
                } else if ahead.peek().is_some_and(|c| c.is_ascii_digit()) {
                    tokens.push(lex_number(&mut chars, false)?);
                } else {
                    chars.next();
                    // Lone '.' — `parse_properties` consumes the
                    // `ident . ident` sequence as a correlated node-property
                    // reference (EqualsNodeProp).
                    tokens.push(Token::Dot);
                }
            }
            '"' | '\'' => {
                let quote = ch;
                chars.next(); // consume opening quote
                let mut s = String::new();
                while let Some(&c) = chars.peek() {
                    if c == quote {
                        chars.next(); // consume closing quote
                        break;
                    }
                    if c == '\\' {
                        chars.next();
                        if let Some(&escaped) = chars.peek() {
                            s.push(match escaped {
                                'n' => '\n',
                                't' => '\t',
                                'r' => '\r',
                                _ => escaped,
                            });
                            chars.next();
                        }
                    } else {
                        s.push(c);
                        chars.next();
                    }
                }
                tokens.push(Token::StringLit(s));
            }
            c if c.is_ascii_digit() => {
                tokens.push(lex_number(&mut chars, false)?);
            }
            '`' => {
                // Backtick-quoted identifier: `programming language`.
                // A doubled backtick is an escaped one, matching the Cypher
                // tokenizer — this lexer reads patterns *re-serialized* by
                // `parser::match_pattern`, so the two escape rules have to be
                // the same or a round-tripped identifier changes meaning.
                chars.next(); // consume opening backtick
                let mut ident = String::new();
                while let Some(&c) = chars.peek() {
                    if c == '`' {
                        chars.next(); // consume the backtick
                        if chars.peek() == Some(&'`') {
                            chars.next(); // …the second of a doubled pair
                            ident.push('`');
                            continue;
                        }
                        break; // it closed the identifier
                    }
                    ident.push(c);
                    chars.next();
                }
                if ident.is_empty() {
                    return Err("Empty backtick identifier".to_string());
                }
                tokens.push(Token::Identifier(ident));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let mut ident = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        ident.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                match ident.to_lowercase().as_str() {
                    "true" => tokens.push(Token::BoolLit(true)),
                    "false" => tokens.push(Token::BoolLit(false)),
                    _ => tokens.push(Token::Identifier(ident)),
                }
            }
            '$' => {
                chars.next(); // consume $
                let mut name = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        name.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if name.is_empty() {
                    return Err("Expected parameter name after '$'".to_string());
                }
                tokens.push(Token::Parameter(name));
            }
            _ => return Err(format!(
                "Unexpected character '{}' in pattern. Valid pattern syntax: (node)-[:EDGE]->(node). \
                Use () for nodes, [] for edges, : for types, {{}} for properties.",
                ch
            )),
        }
    }

    Ok(tokens)
}

/// Builds a `Pattern` AST out of the token stream produced by [`tokenize`]:
/// a sequence of `PatternElement` nodes and edges,
/// `(a:Type {key: val})-[:REL]->(b:Type)`.
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&Token> {
        let token = self.tokens.get(self.pos);
        self.pos += 1;
        token
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        match self.advance() {
            Some(token) if token == expected => Ok(()),
            Some(token) => Err(format!(
                "Syntax error: expected '{}', but found '{}'. Check your pattern syntax.",
                Self::token_to_display(expected),
                Self::token_to_display(token)
            )),
            None => Err(format!(
                "Syntax error: expected '{}', but reached end of pattern. Pattern may be incomplete.",
                Self::token_to_display(expected)
            )),
        }
    }

    fn token_to_display(token: &Token) -> &'static str {
        match token {
            Token::LParen => "(",
            Token::RParen => ")",
            Token::LBracket => "[",
            Token::RBracket => "]",
            Token::LBrace => "{",
            Token::RBrace => "}",
            Token::Colon => ":",
            Token::Comma => ",",
            Token::Dash => "-",
            Token::GreaterThan => ">",
            Token::LessThan => "<",
            Token::Star => "*",
            Token::DotDot => "..",
            Token::Dot => ".",
            Token::Identifier(_) => "identifier",
            Token::StringLit(_) => "string",
            Token::IntLit(_) => "number",
            Token::FloatLit(_) => "decimal",
            Token::BoolLit(_) => "boolean",
            Token::Parameter(_) => "parameter",
            Token::Pipe => "|",
        }
    }

    /// Parse a complete pattern: node (edge node)*
    pub fn parse_pattern(&mut self) -> Result<Pattern, String> {
        let mut elements = Vec::new();
        elements.push(PatternElement::Node(self.parse_node_pattern()?));

        while self.peek().is_some() {
            match self.peek() {
                Some(Token::Dash) | Some(Token::LessThan) => {
                    elements.push(PatternElement::Edge(self.parse_edge_pattern()?));
                    elements.push(PatternElement::Node(self.parse_node_pattern()?));
                }
                _ => break,
            }
        }

        Ok(Pattern { elements })
    }

    /// Consume a name in a label / relationship-type position, which may be
    /// written literally or as a parameter reference (`$label`).
    ///
    /// Returns the text to park in the string slot — the name itself, or the
    /// `$name` placeholder — plus the parameter name when it *was* a
    /// reference. The caller records that reference in the pattern's
    /// `label_params` / `type_params`, which is what the resolver reads; see
    /// [`ParamLabel`] for why the marker is out of band rather than a
    /// spelling inside the string.
    fn expect_label_name(&mut self, context: &str) -> Result<(String, Option<String>), String> {
        match self.advance().cloned() {
            Some(Token::Identifier(name)) => Ok((name, None)),
            Some(Token::Parameter(param)) => Ok((ParamLabel::placeholder(&param), Some(param))),
            _ => Err(context.to_string()),
        }
    }

    /// Parse node pattern: (var:Type {props})
    fn parse_node_pattern(&mut self) -> Result<NodePattern, String> {
        self.expect(&Token::LParen)?;

        let mut variable = None;
        let mut node_type = None;
        let mut extra_labels: Vec<String> = Vec::new();
        let mut properties = None;
        let mut label_params: Vec<ParamLabel> = Vec::new();

        const TYPE_ERR: &str =
            "Expected node type name after ':'. Example: (:Person), (n:Person) or (n:$label)";

        match self.peek() {
            Some(Token::RParen) => {
                // Empty node pattern: ()
            }
            Some(Token::Colon) => {
                // No variable, just type: (:Type) or (:A:B:...)
                self.advance(); // consume :
                let (name, param) = self.expect_label_name(TYPE_ERR)?;
                node_type = Some(name);
                if let Some(param) = param {
                    label_params.push(ParamLabel { slot: 0, param });
                }
            }
            Some(Token::Identifier(_)) => {
                if let Some(Token::Identifier(name)) = self.advance().cloned() {
                    variable = Some(name);
                }
                if let Some(Token::Colon) = self.peek() {
                    self.advance(); // consume :
                    let (name, param) = self.expect_label_name(TYPE_ERR)?;
                    node_type = Some(name);
                    if let Some(param) = param {
                        label_params.push(ParamLabel { slot: 0, param });
                    }
                }
            }
            Some(Token::LBrace) => {
                // Properties only: ({prop: value})
            }
            _ => {}
        }

        // Label alternation `:A|B|C` (Cypher 25 / GQL OR) — mirrors the
        // relationship alternation in `parse_edge_pattern`. Mutually
        // exclusive with the `:A:B` AND-chain below: `:A|B:C` has a
        // precedence this dialect does not implement, so mixing is a parse
        // error rather than a silent commitment.
        let mut alt_labels: Option<Vec<String>> = None;
        while let Some(Token::Pipe) = self.peek() {
            if node_type.is_none() {
                return Err("Label alternation needs a first label: (n:A|B)".to_string());
            }
            self.advance(); // consume |
            let (name, param) = self.expect_label_name(
                "Expected node label name after '|'. Example: (n:Law|Regulation)",
            )?;
            let alts = alt_labels.get_or_insert_with(|| vec![node_type.clone().unwrap()]);
            if !alts.contains(&name) {
                alts.push(name);
            }
            if let Some(param) = param {
                label_params.push(ParamLabel {
                    slot: alts.len() - 1,
                    param,
                });
            }
        }

        // Multi-label suffix: `:A:B:C` collects any extras after the
        // first label. The executor AND-intersects across all labels.
        while let Some(Token::Colon) = self.peek() {
            if alt_labels.is_some() {
                return Err(
                    "Cannot mix label alternation '|' with a ':' label chain in one node \
                     pattern — write (n:A|B) or (n:A:B), or split the predicate into WHERE"
                        .to_string(),
                );
            }
            self.advance(); // consume :
            let (name, param) = self.expect_label_name(
                "Expected node label name after ':'. Example: (n:Person:Manager)",
            )?;
            extra_labels.push(name);
            if let Some(param) = param {
                label_params.push(ParamLabel {
                    slot: extra_labels.len(),
                    param,
                });
            }
        }
        if alt_labels.is_some() && self.peek() == Some(&Token::Pipe) {
            return Err("Unexpected '|' after the label alternation".to_string());
        }

        if let Some(Token::LBrace) = self.peek() {
            properties = Some(self.parse_properties()?);
        }

        self.expect(&Token::RParen)?;

        Ok(NodePattern {
            variable,
            node_type,
            extra_labels,
            alt_labels,
            properties,
            label_params,
        })
    }

    /// Parse edge pattern: -[:TYPE]-> or <-[:TYPE]- or -[:TYPE]-
    /// Also supports variable-length: -[:TYPE*1..3]-> and the openCypher
    /// abbreviated forms without a bracket part: `-->`, `<--`, `--`
    /// (equivalent to -[]->, <-[]-, -[]-).
    fn parse_edge_pattern(&mut self) -> Result<EdgePattern, String> {
        let mut direction = EdgeDirection::Both;
        let mut incoming_start = false;

        if let Some(Token::LessThan) = self.peek() {
            self.advance(); // consume <
            incoming_start = true;
            direction = EdgeDirection::Incoming;
        }

        self.expect(&Token::Dash)?;

        // Abbreviated edge (no bracket part): a second dash immediately
        // follows — `-->` (Dash Dash GreaterThan), `--` (Dash Dash) or
        // `<--` (LessThan Dash Dash, incoming_start already consumed).
        if let Some(Token::Dash) = self.peek() {
            self.advance(); // consume the second -
            if let Some(Token::GreaterThan) = self.peek() {
                self.advance(); // consume >
                if incoming_start {
                    return Err("Invalid edge pattern: cannot have both '<' and '>' arrows. Use --> for outgoing, <-- for incoming, or -- for both directions.".to_string());
                }
                direction = EdgeDirection::Outgoing;
            }
            return Ok(EdgePattern {
                variable: None,
                connection_type: None,
                connection_types: None,
                direction,
                properties: None,
                var_length: None,
                needs_path_info: true,
                skip_target_type_check: false,
                edge_filter: None,
                type_params: Vec::new(),
            });
        }

        // Parse the bracket part: [:TYPE {props}]
        self.expect(&Token::LBracket)?;

        let mut variable = None;
        let mut connection_type = None;
        let mut connection_types: Option<Vec<String>> = None;
        let mut properties = None;
        let mut var_length = None;
        let mut type_params: Vec<ParamLabel> = Vec::new();

        const TYPE_ERR: &str = "Expected connection/edge type after ':'. \
             Example: -[:KNOWS]->, -[e:WORKS_AT]-> or -[:$type]->";

        match self.peek() {
            Some(Token::RBracket) => {
                // Empty edge pattern: []
            }
            Some(Token::Colon) => {
                // No variable, just type: [:TYPE] or [:TYPE1|TYPE2]
                self.advance(); // consume :
                let (name, param) = self.expect_label_name(TYPE_ERR)?;
                connection_type = Some(name);
                if let Some(param) = param {
                    type_params.push(ParamLabel { slot: 0, param });
                }
            }
            Some(Token::Identifier(_)) => {
                if let Some(Token::Identifier(name)) = self.advance().cloned() {
                    variable = Some(name);
                }
                if let Some(Token::Colon) = self.peek() {
                    self.advance(); // consume :
                    let (name, param) = self.expect_label_name(TYPE_ERR)?;
                    connection_type = Some(name);
                    if let Some(param) = param {
                        type_params.push(ParamLabel { slot: 0, param });
                    }
                }
            }
            Some(Token::Star) => {
                // Variable-length without type: [*1..3]
            }
            Some(Token::LBrace) => {
                // Properties only
            }
            _ => {}
        }

        // Pipe-separated types: [:A|B|C] — first type already parsed.
        if connection_type.is_some() {
            if let Some(Token::Pipe) = self.peek() {
                let mut types = vec![connection_type.clone().unwrap()];
                while let Some(Token::Pipe) = self.peek() {
                    self.advance(); // consume |
                    let (name, param) = self.expect_label_name(
                        "Expected connection/edge type after '|'. Example: -[:KNOWS|LIKES]->",
                    )?;
                    types.push(name);
                    if let Some(param) = param {
                        type_params.push(ParamLabel {
                            slot: types.len() - 1,
                            param,
                        });
                    }
                }
                connection_types = Some(types);
            }
        }

        if let Some(Token::Star) = self.peek() {
            var_length = Some(self.parse_var_length()?);
        }

        if let Some(Token::LBrace) = self.peek() {
            properties = Some(self.parse_properties()?);
        }

        self.expect(&Token::RBracket)?;
        self.expect(&Token::Dash)?;

        if let Some(Token::GreaterThan) = self.peek() {
            self.advance(); // consume >
            if incoming_start {
                return Err("Invalid edge pattern: cannot have both '<' and '>' arrows. Use -[]-> for outgoing, <-[]- for incoming, or -[]- for both directions.".to_string());
            }
            direction = EdgeDirection::Outgoing;
        } else if !incoming_start {
            // -[]- without direction is bidirectional
            direction = EdgeDirection::Both;
        }

        Ok(EdgePattern {
            variable,
            connection_type,
            connection_types,
            direction,
            properties,
            var_length,
            needs_path_info: true,
            skip_target_type_check: false,
            edge_filter: None,
            type_params,
        })
    }

    /// Parse variable-length specification: *, *2, *1..3, *..5, *2..
    /// Returns (min_hops, max_hops)
    ///
    /// Open-ended forms (`*`, `*N..`) default the upper bound to
    /// `DEFAULT_MAX_HOPS` as a runaway-query guard — a deliberate,
    /// documented divergence from openCypher's unbounded `*` (recorded in
    /// `tests/api-baselines/cypher-dialect.json` as
    /// `pattern.var_length_default_cap`). An explicit lower bound above the
    /// default (`*11..`) raises the ceiling to that bound so the range is
    /// never silently empty.
    fn parse_var_length(&mut self) -> Result<(usize, usize), String> {
        self.expect(&Token::Star)?;

        // The tokenizer folds a sign into the number it precedes, so `*-1`
        // arrives here as `IntLit(-1)`. An `as usize` cast would turn that
        // into a near-`usize::MAX` hop bound; reject it instead.
        fn hop_count(n: i64) -> Result<usize, String> {
            usize::try_from(n).map_err(|_| {
                format!(
                    "Invalid hop count {} in variable-length path: hop counts must not be \
                     negative. Examples: *2, *1..3, *..5, *1..",
                    n
                )
            })
        }

        const DEFAULT_MAX_HOPS: usize = 10;

        match self.peek() {
            Some(Token::IntLit(_)) => {
                // *N or *N..M or *N..
                let min = if let Some(Token::IntLit(n)) = self.advance().cloned() {
                    hop_count(n)?
                } else {
                    return Err("Expected integer after '*' for variable-length path. Examples: *2, *1..3, *..5, *1..".to_string());
                };

                if let Some(Token::DotDot) = self.peek() {
                    self.advance(); // consume ..
                    if let Some(Token::IntLit(_)) = self.peek() {
                        let max = if let Some(Token::IntLit(n)) = self.advance().cloned() {
                            hop_count(n)?
                        } else {
                            return Err("Expected max hop count after '..'. Examples: *1..3 (1 to 3 hops), *2.. (2 or more hops)".to_string());
                        };
                        if min > max {
                            return Err(format!(
                                "Invalid variable-length range *{}..{}: minimum hop count ({}) \
                                 exceeds maximum ({}). Use *{}..{} instead.",
                                min, max, min, max, max, min
                            ));
                        }
                        Ok((min, max))
                    } else {
                        // *N.. is "N or more", capped at the default —
                        // raised to `min` so the range is never empty.
                        Ok((min, min.max(DEFAULT_MAX_HOPS)))
                    }
                } else {
                    // *N means exactly N hops
                    Ok((min, min))
                }
            }
            Some(Token::DotDot) => {
                // *..M means 1 to M
                self.advance(); // consume ..
                let max = if let Some(Token::IntLit(n)) = self.advance().cloned() {
                    hop_count(n)?
                } else {
                    return Err(
                        "Expected max hop count after '*..'. Example: *..3 means up to 3 hops"
                            .to_string(),
                    );
                };
                Ok((1, max))
            }
            _ => {
                // * alone means 1 or more (up to default max)
                Ok((1, DEFAULT_MAX_HOPS))
            }
        }
    }

    /// Parse properties: {key: value, key2: value2}
    fn parse_properties(&mut self) -> Result<HashMap<String, PropertyMatcher>, String> {
        self.expect(&Token::LBrace)?;
        let mut props = HashMap::new();

        loop {
            match self.peek() {
                Some(Token::RBrace) => {
                    self.advance();
                    break;
                }
                Some(Token::Identifier(_)) => {
                    let key = if let Some(Token::Identifier(k)) = self.advance().cloned() {
                        k
                    } else {
                        return Err("Expected property key in properties block. Example: {name: 'Alice', age: 30}".to_string());
                    };

                    self.expect(&Token::Colon)?;

                    if let Some(Token::Parameter(_)) = self.peek() {
                        if let Some(Token::Parameter(name)) = self.advance().cloned() {
                            props.insert(key, PropertyMatcher::EqualsParam(name));
                        }
                    } else if let Some(Token::Identifier(_)) = self.peek() {
                        if let Some(Token::Identifier(name)) = self.advance().cloned() {
                            if let Some(Token::Dot) = self.peek() {
                                // `var.prop` → correlated node-property reference,
                                // e.g. WITH collect(x)[0] AS first
                                //      MATCH (b {id: first.id})
                                self.advance(); // consume '.'
                                if let Some(Token::Identifier(prop)) = self.advance().cloned() {
                                    props.insert(
                                        key,
                                        PropertyMatcher::EqualsNodeProp { var: name, prop },
                                    );
                                } else {
                                    return Err(
                                        "Expected a property name after '.' in inline map value \
                                         (e.g. {id: other.id})"
                                            .to_string(),
                                    );
                                }
                            } else {
                                // Bare identifier → variable reference from outer
                                // scope, e.g. WITH 'Oslo' AS city MATCH (n {city: city})
                                props.insert(key, PropertyMatcher::EqualsVar(name));
                            }
                        }
                    } else {
                        let value = self.parse_value()?;
                        props.insert(key, PropertyMatcher::Equals(value));
                    }

                    if let Some(Token::Comma) = self.peek() {
                        self.advance();
                    }
                }
                _ => return Err("Expected property key or '}' to close properties block. Example: {name: 'Alice'}".to_string()),
            }
        }

        Ok(props)
    }

    /// Parse a value (string, int, float, bool)
    ///
    /// The tokenizer folds a sign that is adjacent to its digits into the
    /// literal (`{x: -1}` → `IntLit(-1)`). A separated sign only reaches
    /// here from the EXISTS-subquery pattern re-serializer, which joins
    /// tokens with a space (`{x: - 1}`), so the `Dash` arm negates the
    /// following literal. A literal that is *already* negative there means
    /// a doubled sign (`--1`, `- -1`) and stays an error.
    fn parse_value(&mut self) -> Result<Value, String> {
        match self.advance().cloned() {
            Some(Token::StringLit(s)) => Ok(Value::String(s)),
            Some(Token::IntLit(i)) => Ok(Value::Int64(i)),
            Some(Token::FloatLit(f)) => Ok(Value::Float64(f)),
            Some(Token::BoolLit(b)) => Ok(Value::Boolean(b)),
            Some(Token::Dash) => match self.advance().cloned() {
                Some(Token::IntLit(i)) if i >= 0 => Ok(Value::Int64(-i)),
                Some(Token::FloatLit(f)) if !f.is_sign_negative() => Ok(Value::Float64(-f)),
                Some(token) => Err(format!(
                    "Expected a numeric literal after '-' in an inline map value \
                     (e.g. {{temp: -1}}), got {:?}",
                    token
                )),
                None => Err(
                    "Expected a numeric literal after '-' in an inline map value \
                     (e.g. {temp: -1}), got end of input"
                        .to_string(),
                ),
            },
            Some(token) => Err(format!("Expected value, got {:?}", token)),
            None => Err("Expected value, got end of input".to_string()),
        }
    }
}

pub fn parse_pattern(input: &str) -> Result<Pattern, String> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens);
    let pattern = parser.parse_pattern()?;
    // The whole input must be one pattern. Pre-fix, trailing tokens were
    // silently discarded, so `MATCH (n) bogus tokens` — including a typo'd
    // keyword (`RETRUN n`) — executed as `MATCH (n)` with no error and a
    // different meaning than the user wrote.
    if let Some(tok) = parser.peek() {
        return Err(format!(
            "unexpected trailing input after pattern: {tok:?} (in {input:?})"
        ));
    }
    Ok(pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_simple() {
        let tokens = tokenize("(a:Person)").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::LParen,
                Token::Identifier("a".to_string()),
                Token::Colon,
                Token::Identifier("Person".to_string()),
                Token::RParen,
            ]
        );
    }

    #[test]
    fn quoting_predicate_agrees_with_the_tokenizer() {
        // `bare_word_needs_quoting` tells the pattern re-serializer which
        // names it may write bare. If it ever disagrees with this tokenizer,
        // a name changes meaning in transit — the create-then-match asymmetry
        // that made a backticked `TRUE` label unmatchable. Both directions
        // are checked, so the predicate can neither under- nor over-claim.
        for word in [
            "true", "TRUE", "True", "false", "FALSE", "fAlSe", "null", "NULL", "Person", "order",
            "contains", "x", "_x", "t1", "$label", "$", "$1",
        ] {
            // A word the tokenizer *rejects* outright (a bare `$`) also fails
            // to lex as itself, so the predicate must demand quoting for it.
            let lexes_as_itself = matches!(
                tokenize(word).as_deref(),
                Ok([Token::Identifier(s)]) if s == word
            );
            assert_eq!(
                bare_word_needs_quoting(word),
                !lexes_as_itself,
                "{word:?}: the quoting predicate and the tokenizer disagree"
            );
            // And the escape always works, whatever the verdict.
            assert_eq!(
                tokenize(&format!("`{word}`")).unwrap(),
                vec![Token::Identifier(word.to_string())]
            );
        }
    }

    #[test]
    fn test_tokenize_edge() {
        let tokens = tokenize("-[:KNOWS]->").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::Dash,
                Token::LBracket,
                Token::Colon,
                Token::Identifier("KNOWS".to_string()),
                Token::RBracket,
                Token::Dash,
                Token::GreaterThan,
            ]
        );
    }

    #[test]
    fn test_tokenize_properties() {
        let tokens = tokenize("{name: \"Alice\", age: 30}").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::LBrace,
                Token::Identifier("name".to_string()),
                Token::Colon,
                Token::StringLit("Alice".to_string()),
                Token::Comma,
                Token::Identifier("age".to_string()),
                Token::Colon,
                Token::IntLit(30),
                Token::RBrace,
            ]
        );
    }

    #[test]
    fn test_parse_simple_node() {
        let pattern = parse_pattern("(p:Person)").unwrap();
        assert_eq!(pattern.elements.len(), 1);
        if let PatternElement::Node(np) = &pattern.elements[0] {
            assert_eq!(np.variable, Some("p".to_string()));
            assert_eq!(np.node_type, Some("Person".to_string()));
        } else {
            panic!("Expected node pattern");
        }
    }

    #[test]
    fn test_parse_multi_label_node() {
        let pattern = parse_pattern("(a:Person:Director)").unwrap();
        if let PatternElement::Node(np) = &pattern.elements[0] {
            assert_eq!(np.node_type, Some("Person".to_string()));
            assert_eq!(np.extra_labels, vec!["Director".to_string()]);
        } else {
            panic!("Expected node pattern");
        }
    }

    #[test]
    fn test_parse_three_labels() {
        let pattern = parse_pattern("(n:Animal:Pet:Dog)").unwrap();
        if let PatternElement::Node(np) = &pattern.elements[0] {
            assert_eq!(np.node_type, Some("Animal".to_string()));
            assert_eq!(np.extra_labels, vec!["Pet".to_string(), "Dog".to_string()]);
        } else {
            panic!("Expected node pattern");
        }
    }

    #[test]
    fn test_parse_single_label_has_empty_extras() {
        let pattern = parse_pattern("(p:Person)").unwrap();
        if let PatternElement::Node(np) = &pattern.elements[0] {
            assert!(np.extra_labels.is_empty());
        } else {
            panic!("Expected node pattern");
        }
    }

    #[test]
    fn test_parse_node_with_properties() {
        let pattern = parse_pattern("(p:Person {name: \"Alice\"})").unwrap();
        if let PatternElement::Node(np) = &pattern.elements[0] {
            assert!(np.properties.is_some());
            let props = np.properties.as_ref().unwrap();
            assert!(props.contains_key("name"));
        } else {
            panic!("Expected node pattern");
        }
    }

    #[test]
    fn test_parse_single_hop() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS]->(b:Person)").unwrap();
        assert_eq!(pattern.elements.len(), 3);

        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            assert_eq!(ep.connection_type, Some("KNOWS".to_string()));
            assert_eq!(ep.direction, EdgeDirection::Outgoing);
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_incoming_edge() {
        let pattern = parse_pattern("(a:Person)<-[:KNOWS]-(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            assert_eq!(ep.direction, EdgeDirection::Incoming);
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_bidirectional_edge() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS]-(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            assert_eq!(ep.direction, EdgeDirection::Both);
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_multi_hop() {
        let pattern =
            parse_pattern("(a:Person)-[:KNOWS]->(b:Person)-[:WORKS_AT]->(c:Company)").unwrap();
        assert_eq!(pattern.elements.len(), 5);
    }

    #[test]
    fn test_parse_anonymous_node() {
        let pattern = parse_pattern("(:Person)").unwrap();
        if let PatternElement::Node(np) = &pattern.elements[0] {
            assert_eq!(np.variable, None);
            assert_eq!(np.node_type, Some("Person".to_string()));
        } else {
            panic!("Expected node pattern");
        }
    }

    #[test]
    fn test_parse_empty_node() {
        let pattern = parse_pattern("()").unwrap();
        if let PatternElement::Node(np) = &pattern.elements[0] {
            assert_eq!(np.variable, None);
            assert_eq!(np.node_type, None);
        } else {
            panic!("Expected node pattern");
        }
    }

    // Variable-length path tests
    #[test]
    fn test_tokenize_var_length() {
        let tokens = tokenize("-[:KNOWS*1..3]->").unwrap();
        assert!(tokens.contains(&Token::Star));
        assert!(tokens.contains(&Token::DotDot));
        assert!(tokens.contains(&Token::IntLit(1)));
        assert!(tokens.contains(&Token::IntLit(3)));
    }

    #[test]
    fn test_parse_var_length_exact() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS*2]->(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            assert_eq!(ep.var_length, Some((2, 2)));
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_var_length_range() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS*1..3]->(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            assert_eq!(ep.var_length, Some((1, 3)));
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_var_length_min_only() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS*2..]->(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            // *2.. means 2 to default max (10)
            assert_eq!(ep.var_length, Some((2, 10)));
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_var_length_max_only() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS*..5]->(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            assert_eq!(ep.var_length, Some((1, 5)));
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_var_length_star_only() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS*]->(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            // * alone means 1 to default max (10)
            assert_eq!(ep.var_length, Some((1, 10)));
        } else {
            panic!("Expected edge pattern");
        }
    }

    #[test]
    fn test_parse_normal_edge_no_var_length() {
        let pattern = parse_pattern("(a:Person)-[:KNOWS]->(b:Person)").unwrap();
        if let PatternElement::Edge(ep) = &pattern.elements[1] {
            assert_eq!(ep.var_length, None);
        } else {
            panic!("Expected edge pattern");
        }
    }

    // Abbreviated (bracketless) edge forms: -->, --, <--

    fn abbreviated_edge(pattern_str: &str) -> EdgePattern {
        let pattern = parse_pattern(pattern_str).unwrap();
        assert_eq!(pattern.elements.len(), 3);
        match &pattern.elements[1] {
            PatternElement::Edge(ep) => ep.clone(),
            _ => panic!("Expected edge pattern"),
        }
    }

    #[test]
    fn test_parse_abbreviated_outgoing() {
        let ep = abbreviated_edge("(a)-->(b)");
        assert_eq!(ep.direction, EdgeDirection::Outgoing);
        assert_eq!(ep.connection_type, None);
        assert_eq!(ep.variable, None);
        assert_eq!(ep.var_length, None);
    }

    #[test]
    fn test_parse_abbreviated_undirected() {
        let ep = abbreviated_edge("(a)--(b)");
        assert_eq!(ep.direction, EdgeDirection::Both);
        assert_eq!(ep.connection_type, None);
    }

    #[test]
    fn test_parse_abbreviated_incoming() {
        let ep = abbreviated_edge("(a)<--(b)");
        assert_eq!(ep.direction, EdgeDirection::Incoming);
        assert_eq!(ep.connection_type, None);
    }

    #[test]
    fn test_parse_abbreviated_multi_hop() {
        let pattern = parse_pattern("(a)-->(b)--(c)<--(d)").unwrap();
        assert_eq!(pattern.elements.len(), 7);
    }

    #[test]
    fn test_parse_abbreviated_double_arrow_rejected() {
        // <--> is invalid — both arrowheads.
        assert!(parse_pattern("(a)<-->(b)").is_err());
    }

    #[test]
    fn test_parse_single_dash_still_rejected() {
        // `(a)-(b)` is not a pattern edge (a lone dash is subtraction in
        // expression positions and invalid in patterns).
        assert!(parse_pattern("(a)-(b)").is_err());
    }

    // Negative inline-map literals: `MATCH (n {x: -1})`.

    fn node_props(pattern_str: &str) -> HashMap<String, PropertyMatcher> {
        let pattern = parse_pattern(pattern_str).unwrap();
        match &pattern.elements[0] {
            PatternElement::Node(np) => np.properties.clone().expect("node properties"),
            _ => panic!("Expected node pattern"),
        }
    }

    fn edge_props(pattern_str: &str) -> HashMap<String, PropertyMatcher> {
        let pattern = parse_pattern(pattern_str).unwrap();
        match &pattern.elements[1] {
            PatternElement::Edge(ep) => ep.properties.clone().expect("edge properties"),
            _ => panic!("Expected edge pattern"),
        }
    }

    fn equals(props: &HashMap<String, PropertyMatcher>, key: &str) -> Value {
        match props.get(key) {
            Some(PropertyMatcher::Equals(v)) => v.clone(),
            other => panic!("Expected an Equals matcher for {}, got {:?}", key, other),
        }
    }

    #[test]
    fn test_parse_negative_int_in_node_map() {
        let props = node_props("(n:Reading {temp: -1})");
        assert_eq!(equals(&props, "temp"), Value::Int64(-1));
    }

    #[test]
    fn test_parse_negative_float_in_node_map() {
        let props = node_props("(n:Reading {delta: -1.5})");
        assert_eq!(equals(&props, "delta"), Value::Float64(-1.5));
    }

    #[test]
    fn test_parse_negative_literals_in_edge_map() {
        let props = edge_props("(a:P)-[r:DELTA {temp: -1, change: -1.5}]->(b:P)");
        assert_eq!(equals(&props, "temp"), Value::Int64(-1));
        assert_eq!(equals(&props, "change"), Value::Float64(-1.5));
    }

    #[test]
    fn test_tokenize_signed_int_is_one_literal() {
        // The sign must be consumed as part of the literal. Lexing the
        // magnitude first and negating afterwards cannot represent
        // i64::MIN — 9223372036854775808 does not fit in an i64 — so this
        // is the test that catches a parse-positive-then-negate fix.
        let tokens = tokenize("{x: -9223372036854775808}").unwrap();
        assert!(
            tokens.contains(&Token::IntLit(i64::MIN)),
            "expected a single IntLit(i64::MIN) token, got {:?}",
            tokens
        );
        assert!(
            !tokens.contains(&Token::Dash),
            "sign left unconsumed: {:?}",
            tokens
        );
    }

    #[test]
    fn test_parse_i64_min_in_node_map() {
        let props = node_props("(n:Reading {temp: -9223372036854775808})");
        assert_eq!(equals(&props, "temp"), Value::Int64(i64::MIN));
    }

    #[test]
    fn test_parse_negative_literal_with_space_after_dash() {
        // The EXISTS-subquery pattern re-serializer joins tokens with a
        // space, so a negative literal reaches this parser as `- 1`.
        let props = node_props("( n:Reading { temp : - 1 , delta : - 1.5 } )");
        assert_eq!(equals(&props, "temp"), Value::Int64(-1));
        assert_eq!(equals(&props, "delta"), Value::Float64(-1.5));
    }

    #[test]
    fn test_parse_malformed_dash_values_still_rejected() {
        for bad in [
            "(n:Reading {temp: -})",
            "(n:Reading {temp: --1})",
            "(n:Reading {temp: - -1})",
            "(n:Reading {temp: -'x'})",
        ] {
            assert!(
                parse_pattern(bad).is_err(),
                "expected {} to stay a parse error",
                bad
            );
        }
    }

    #[test]
    fn test_negative_hop_counts_rejected() {
        // The tokenizer folds the sign into the literal, so the hop-count
        // parser must reject it rather than cast it to a huge usize.
        for bad in [
            "(a)-[:K*-1]->(b)",
            "(a)-[:K*1..-3]->(b)",
            "(a)-[:K*..-3]->(b)",
        ] {
            let err = parse_pattern(bad).unwrap_err();
            assert!(
                err.contains("hop count"),
                "expected a hop-count error for {}, got: {}",
                bad,
                err
            );
        }
    }

    #[test]
    fn test_structural_dash_unaffected_by_negative_literals() {
        // A relationship whose dashes are structural still parses, including
        // when both endpoints and the edge carry negative inline literals.
        let pattern =
            parse_pattern("(a:P {temp: -1})-[r:DELTA {change: -1.5}]->(b:P {temp: -2})").unwrap();
        assert_eq!(pattern.elements.len(), 3);
        match &pattern.elements[1] {
            PatternElement::Edge(ep) => {
                assert_eq!(ep.direction, EdgeDirection::Outgoing);
                assert_eq!(ep.connection_type, Some("DELTA".to_string()));
                assert_eq!(ep.var_length, None);
            }
            _ => panic!("Expected edge pattern"),
        }
        // Bare and incoming forms keep their structural dashes.
        assert_eq!(parse_pattern("(a)-->(b)").unwrap().elements.len(), 3);
        assert_eq!(parse_pattern("(a)<-[:K]-(b)").unwrap().elements.len(), 3);
    }
}
