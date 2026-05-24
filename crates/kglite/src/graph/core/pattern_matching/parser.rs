// Parser — tokenizes and parses Cypher-like pattern strings into a Pattern AST.

use crate::datatypes::values::Value;
use std::collections::HashMap;

use super::pattern::{
    EdgeDirection, EdgePattern, NodePattern, Pattern, PatternElement, PropertyMatcher,
};

// ============================================================================
// Tokenizer
// ============================================================================

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
    Pipe,        // | (for multi-type edges: [:A|B])
    Identifier(String),
    StringLit(String),
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    Parameter(String), // $param_name
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
                tokens.push(Token::Dash);
                chars.next();
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
                // Check for '..' (range operator)
                chars.next();
                if chars.peek() == Some(&'.') {
                    chars.next();
                    tokens.push(Token::DotDot);
                } else if chars.peek().is_some_and(|c| c.is_ascii_digit()) {
                    // It's a float starting with '.'
                    let mut num_str = String::from("0.");
                    while let Some(&c) = chars.peek() {
                        if c.is_ascii_digit() {
                            num_str.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    tokens.push(Token::FloatLit(
                        num_str.parse().map_err(|_| format!("Invalid float: {}", num_str))?
                    ));
                } else {
                    return Err("Unexpected single '.', expected '..' or a digit".to_string());
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
                let mut num_str = String::new();
                let mut has_dot = false;
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() {
                        num_str.push(c);
                        chars.next();
                    } else if c == '.' && !has_dot {
                        // Peek ahead to check if this is '..' (range operator)
                        // Clone the iterator to peek ahead without consuming
                        let mut peek_chars = chars.clone();
                        peek_chars.next(); // skip the first '.'
                        if peek_chars.peek() == Some(&'.') {
                            // This is '..', stop here and don't include the dot
                            break;
                        }
                        // It's a decimal point for a float
                        has_dot = true;
                        num_str.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if has_dot {
                    tokens.push(Token::FloatLit(
                        num_str.parse().map_err(|_| format!("Invalid float: {}", num_str))?
                    ));
                } else {
                    tokens.push(Token::IntLit(
                        num_str.parse().map_err(|_| format!("Invalid integer: {}", num_str))?
                    ));
                }
            }
            '`' => {
                // Backtick-quoted identifier: `programming language`
                chars.next(); // consume opening backtick
                let mut ident = String::new();
                while let Some(&c) = chars.peek() {
                    if c == '`' {
                        chars.next(); // consume closing backtick
                        break;
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
                // Check for boolean literals
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

// ============================================================================
// Parser
// ============================================================================

/// Parses Cypher-like pattern strings into a `Pattern` AST.
///
/// Tokenizes the input, then builds a sequence of `PatternElement`
/// nodes and edges: `(a:Type {key: val})-[:REL]->(b:Type)`.
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

        // Must start with a node pattern
        elements.push(PatternElement::Node(self.parse_node_pattern()?));

        // Parse edge-node pairs
        while self.peek().is_some() {
            // Check for edge pattern (starts with - or <)
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

    /// Parse node pattern: (var:Type {props})
    fn parse_node_pattern(&mut self) -> Result<NodePattern, String> {
        self.expect(&Token::LParen)?;

        let mut variable = None;
        let mut node_type = None;
        let mut properties = None;

        // Check what comes next
        match self.peek() {
            Some(Token::RParen) => {
                // Empty node pattern: ()
            }
            Some(Token::Colon) => {
                // No variable, just type: (:Type)
                self.advance(); // consume :
                if let Some(Token::Identifier(name)) = self.advance().cloned() {
                    node_type = Some(name);
                } else {
                    return Err(
                        "Expected node type name after ':'. Example: (:Person) or (n:Person)"
                            .to_string(),
                    );
                }
            }
            Some(Token::Identifier(_)) => {
                // Variable name
                if let Some(Token::Identifier(name)) = self.advance().cloned() {
                    variable = Some(name);
                }
                // Check for type
                if let Some(Token::Colon) = self.peek() {
                    self.advance(); // consume :
                    if let Some(Token::Identifier(name)) = self.advance().cloned() {
                        node_type = Some(name);
                    } else {
                        return Err(
                            "Expected node type name after ':'. Example: (:Person) or (n:Person)"
                                .to_string(),
                        );
                    }
                }
            }
            Some(Token::LBrace) => {
                // Properties only: ({prop: value})
            }
            _ => {}
        }

        // Check for properties
        if let Some(Token::LBrace) = self.peek() {
            properties = Some(self.parse_properties()?);
        }

        self.expect(&Token::RParen)?;

        Ok(NodePattern {
            variable,
            node_type,
            properties,
        })
    }

    /// Parse edge pattern: -[:TYPE]-> or <-[:TYPE]- or -[:TYPE]-
    /// Also supports variable-length: -[:TYPE*1..3]->
    fn parse_edge_pattern(&mut self) -> Result<EdgePattern, String> {
        let mut direction = EdgeDirection::Both;
        let mut incoming_start = false;

        // Check for incoming arrow start: <-
        if let Some(Token::LessThan) = self.peek() {
            self.advance(); // consume <
            incoming_start = true;
            direction = EdgeDirection::Incoming;
        }

        self.expect(&Token::Dash)?;

        // Parse the bracket part: [:TYPE {props}]
        self.expect(&Token::LBracket)?;

        let mut variable = None;
        let mut connection_type = None;
        let mut connection_types: Option<Vec<String>> = None;
        let mut properties = None;
        let mut var_length = None;

        // Check what comes next
        match self.peek() {
            Some(Token::RBracket) => {
                // Empty edge pattern: []
            }
            Some(Token::Colon) => {
                // No variable, just type: [:TYPE] or [:TYPE1|TYPE2]
                self.advance(); // consume :
                if let Some(Token::Identifier(name)) = self.advance().cloned() {
                    connection_type = Some(name);
                } else {
                    return Err("Expected connection/edge type after ':'. Example: -[:KNOWS]-> or -[e:WORKS_AT]->".to_string());
                }
            }
            Some(Token::Identifier(_)) => {
                // Variable name
                if let Some(Token::Identifier(name)) = self.advance().cloned() {
                    variable = Some(name);
                }
                // Check for type
                if let Some(Token::Colon) = self.peek() {
                    self.advance(); // consume :
                    if let Some(Token::Identifier(name)) = self.advance().cloned() {
                        connection_type = Some(name);
                    } else {
                        return Err("Expected connection/edge type after ':'. Example: -[:KNOWS]-> or -[e:WORKS_AT]->".to_string());
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

        // Handle pipe-separated types: [:A|B|C]
        // After parsing the first type, consume any |TYPE continuations
        if connection_type.is_some() {
            if let Some(Token::Pipe) = self.peek() {
                let mut types = vec![connection_type.clone().unwrap()];
                while let Some(Token::Pipe) = self.peek() {
                    self.advance(); // consume |
                    if let Some(Token::Identifier(name)) = self.advance().cloned() {
                        types.push(name);
                    } else {
                        return Err(
                            "Expected connection/edge type after '|'. Example: -[:KNOWS|LIKES]->"
                                .to_string(),
                        );
                    }
                }
                connection_types = Some(types);
            }
        }

        // Check for variable-length marker: *
        if let Some(Token::Star) = self.peek() {
            var_length = Some(self.parse_var_length()?);
        }

        // Check for properties
        if let Some(Token::LBrace) = self.peek() {
            properties = Some(self.parse_properties()?);
        }

        self.expect(&Token::RBracket)?;
        self.expect(&Token::Dash)?;

        // Check for outgoing arrow end: ->
        if let Some(Token::GreaterThan) = self.peek() {
            self.advance(); // consume >
            if incoming_start {
                // <-[]-> is invalid
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
        })
    }

    /// Parse variable-length specification: *, *2, *1..3, *..5, *2..
    /// Returns (min_hops, max_hops)
    fn parse_var_length(&mut self) -> Result<(usize, usize), String> {
        self.expect(&Token::Star)?;

        const DEFAULT_MAX_HOPS: usize = 10; // Reasonable limit to prevent runaway queries

        // Check what follows the *
        match self.peek() {
            Some(Token::IntLit(_)) => {
                // *N or *N..M or *N..
                let min = if let Some(Token::IntLit(n)) = self.advance().cloned() {
                    n as usize
                } else {
                    return Err("Expected integer after '*' for variable-length path. Examples: *2, *1..3, *..5, *1..".to_string());
                };

                // Check for range
                if let Some(Token::DotDot) = self.peek() {
                    self.advance(); // consume ..
                                    // Check for max
                    if let Some(Token::IntLit(_)) = self.peek() {
                        let max = if let Some(Token::IntLit(n)) = self.advance().cloned() {
                            n as usize
                        } else {
                            return Err("Expected max hop count after '..'. Examples: *1..3 (1 to 3 hops), *2.. (2 or more hops)".to_string());
                        };
                        Ok((min, max))
                    } else {
                        // *N.. means N to default max
                        Ok((min, DEFAULT_MAX_HOPS))
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
                    n as usize
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
                    // Parse key: value
                    let key = if let Some(Token::Identifier(k)) = self.advance().cloned() {
                        k
                    } else {
                        return Err("Expected property key in properties block. Example: {name: 'Alice', age: 30}".to_string());
                    };

                    self.expect(&Token::Colon)?;

                    // Check if next token is a parameter reference
                    if let Some(Token::Parameter(_)) = self.peek() {
                        if let Some(Token::Parameter(name)) = self.advance().cloned() {
                            props.insert(key, PropertyMatcher::EqualsParam(name));
                        }
                    } else if let Some(Token::Identifier(_)) = self.peek() {
                        // Bare identifier → variable reference from outer scope
                        // e.g. WITH "Oslo" AS city MATCH (n {city: city})
                        if let Some(Token::Identifier(name)) = self.advance().cloned() {
                            props.insert(key, PropertyMatcher::EqualsVar(name));
                        }
                    } else {
                        let value = self.parse_value()?;
                        props.insert(key, PropertyMatcher::Equals(value));
                    }

                    // Check for comma or end
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
    fn parse_value(&mut self) -> Result<Value, String> {
        match self.advance().cloned() {
            Some(Token::StringLit(s)) => Ok(Value::String(s)),
            Some(Token::IntLit(i)) => Ok(Value::Int64(i)),
            Some(Token::FloatLit(f)) => Ok(Value::Float64(f)),
            Some(Token::BoolLit(b)) => Ok(Value::Boolean(b)),
            Some(token) => Err(format!("Expected value, got {:?}", token)),
            None => Err("Expected value, got end of input".to_string()),
        }
    }
}

pub fn parse_pattern(input: &str) -> Result<Pattern, String> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens);
    parser.parse_pattern()
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
}
