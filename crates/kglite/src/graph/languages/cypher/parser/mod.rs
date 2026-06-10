//! Cypher parser — delegates MATCH patterns to
//! `crate::graph::core::pattern_matching::parse_pattern`.
//!
//! Split (Phase 9):
//! - [`match_pattern`] — MATCH / OPTIONAL MATCH, pattern extraction, EXISTS
//! - [`predicate`] — WHERE predicate chain (AND / OR / XOR / NOT / comparisons)
//! - [`expression`] — expressions (arithmetic, function calls, CASE, list ops)
//! - [`clauses`] — RETURN / WITH / ORDER BY / LIMIT / SKIP / UNWIND / UNION /
//!   CREATE / SET / DELETE / REMOVE / MERGE / CALL
//!
//! Each submodule adds another `impl CypherParser` block; PyO3-style,
//! Rust merges them at codegen.

use super::ast::*;
use super::tokenizer::{keyword_name_token, token_to_keyword_name, CypherToken};
#[cfg(test)]
use crate::datatypes::values::Value;
use crate::error::KgError;

pub mod clauses;
pub mod expression;
pub mod match_pattern;
pub mod predicate;

/// Tokenizes and parses Cypher query strings into a `CypherQuery` AST.
///
/// Handles the full Cypher clause set: MATCH, WHERE, RETURN, WITH,
/// ORDER BY, LIMIT, SKIP, CREATE, SET, DELETE, MERGE, REMOVE, UNWIND, UNION.
/// Uses a token-based recursive descent approach.
pub struct CypherParser {
    tokens: Vec<CypherToken>,
    pos: usize,
}

impl CypherParser {
    pub fn new(tokens: Vec<CypherToken>) -> Self {
        CypherParser { tokens, pos: 0 }
    }

    // ========================================================================
    // Token Navigation
    // ========================================================================

    pub(super) fn peek(&self) -> Option<&CypherToken> {
        self.tokens.get(self.pos)
    }

    pub(super) fn peek_at(&self, offset: usize) -> Option<&CypherToken> {
        self.tokens.get(self.pos + offset)
    }

    pub(super) fn advance(&mut self) -> Option<&CypherToken> {
        let token = self.tokens.get(self.pos);
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    pub(super) fn expect(&mut self, expected: &CypherToken) -> Result<(), String> {
        match self.peek() {
            Some(t) if t == expected => {
                self.advance();
                Ok(())
            }
            Some(t) => Err(format!("Expected {:?}, found {:?}", expected, t)),
            None => Err(format!("Expected {:?}, but reached end of query", expected)),
        }
    }

    pub(super) fn has_tokens(&self) -> bool {
        self.pos < self.tokens.len()
    }

    /// Check if current position matches a keyword
    pub(super) fn check(&self, token: &CypherToken) -> bool {
        self.peek() == Some(token)
    }

    /// Consume the next token as an alias name (after AS).
    /// Accepts identifiers and reserved keywords (e.g. `AS optional`, `AS type`).
    pub(super) fn try_consume_alias_name(&mut self) -> Result<String, String> {
        match self.advance().cloned() {
            Some(CypherToken::Identifier(name)) => Ok(name),
            Some(ref token) => token_to_keyword_name(token)
                .ok_or_else(|| format!("Expected alias name after AS, got {:?}", token)),
            None => Err("Expected alias name after AS".to_string()),
        }
    }

    /// Consume the next token as a NAME — a node label, relationship type, or
    /// property key. Accepts an identifier verbatim, or a soft-reservable
    /// keyword via `keyword_name_token` (KG-2: `[:CONTAINS]`, `(:CONTAINS)`,
    /// `{contains: 1}`). `context` names the position for the error message,
    /// preserving the original "Expected <X>" wording. Case-preserving: a
    /// keyword name round-trips as its canonical uppercase word.
    pub(super) fn expect_name(&mut self, context: &str) -> Result<String, String> {
        match self.advance().cloned() {
            Some(CypherToken::Identifier(name)) => Ok(name),
            Some(ref token) => keyword_name_token(token)
                .map(str::to_string)
                .ok_or_else(|| format!("Expected {}, got {:?}", context, token)),
            None => Err(format!("Expected {}", context)),
        }
    }

    /// Check if we're at a clause boundary (start of a new clause)
    pub(super) fn at_clause_boundary(&self) -> bool {
        match self.peek() {
            Some(CypherToken::Where)
            | Some(CypherToken::Return)
            | Some(CypherToken::With)
            | Some(CypherToken::Limit)
            | Some(CypherToken::Skip)
            | Some(CypherToken::Unwind)
            | Some(CypherToken::Union)
            | Some(CypherToken::Intersect)
            | Some(CypherToken::Except)
            | Some(CypherToken::Create)
            | Some(CypherToken::Set)
            | Some(CypherToken::Delete)
            | Some(CypherToken::Detach)
            | Some(CypherToken::Merge)
            | Some(CypherToken::Remove)
            | Some(CypherToken::On)
            | Some(CypherToken::Call)
            | Some(CypherToken::Yield)
            | Some(CypherToken::Having) => true,
            Some(CypherToken::Match) => true,
            Some(CypherToken::Optional) => {
                // OPTIONAL MATCH
                self.peek_at(1) == Some(&CypherToken::Match)
            }
            Some(CypherToken::Order) => {
                // ORDER BY
                self.peek_at(1) == Some(&CypherToken::By)
            }
            None => true,
            _ => false,
        }
    }

    // ========================================================================
    // Top-Level Query Parser
    // ========================================================================

    pub fn parse_query(&mut self) -> Result<CypherQuery, String> {
        // Check for EXPLAIN or PROFILE prefix
        let mut explain = false;
        let mut profile = false;
        if self.check(&CypherToken::Explain) {
            self.advance();
            explain = true;
        } else if self.check(&CypherToken::Profile) {
            self.advance();
            profile = true;
        }

        let (clauses, output_format) = self.parse_clause_sequence(false)?;

        if clauses.is_empty() {
            return Err("Empty query".to_string());
        }

        Ok(CypherQuery {
            clauses,
            explain,
            profile,
            output_format,
        })
    }

    /// Parse a sequence of clauses into the body of a query.
    ///
    /// When `end_at_rbrace` is `false` the loop runs until end-of-input
    /// (the top-level query). When `true` it stops at — and leaves
    /// unconsumed — the closing `}` of a `CALL { ... }` subquery body; the
    /// caller (`parse_call_subquery`) is responsible for consuming that
    /// brace. Nested `{ ... }` (map literals, nested `CALL {}`) are handled
    /// by the per-clause parsers, which consume their own braces in
    /// balanced pairs — so a `RBrace` seen *at clause-boundary level* here
    /// is unambiguously the subquery terminator.
    ///
    /// Returns the parsed clauses plus the trailing `OutputFormat` (only a
    /// top-level `FORMAT CSV` sets it to `Csv`; subquery bodies reject
    /// `FORMAT`).
    pub(super) fn parse_clause_sequence(
        &mut self,
        end_at_rbrace: bool,
    ) -> Result<(Vec<Clause>, OutputFormat), String> {
        let mut clauses = Vec::new();

        while self.has_tokens() {
            // Closing brace of a CALL { ... } body — stop, leave it for the caller.
            if end_at_rbrace && self.check(&CypherToken::RBrace) {
                break;
            }

            // Skip semicolons between statements
            if self.check(&CypherToken::Semicolon) {
                self.advance();
                continue;
            }

            match self.peek() {
                Some(CypherToken::Match) => {
                    clauses.push(self.parse_match_clause(false)?);
                }
                Some(CypherToken::Optional) => {
                    // Check for OPTIONAL MATCH
                    if self.peek_at(1) == Some(&CypherToken::Match) {
                        self.advance(); // consume OPTIONAL
                        clauses.push(self.parse_match_clause(true)?);
                    } else {
                        return Err("Expected MATCH after OPTIONAL".to_string());
                    }
                }
                Some(CypherToken::Where) => {
                    clauses.push(self.parse_where_clause()?);
                }
                Some(CypherToken::Return) => {
                    clauses.push(self.parse_return_clause()?);
                }
                Some(CypherToken::With) => {
                    clauses.push(self.parse_with_clause()?);
                }
                Some(CypherToken::Order) => {
                    clauses.push(self.parse_order_by_clause()?);
                }
                Some(CypherToken::Limit) => {
                    clauses.push(self.parse_limit_clause()?);
                }
                Some(CypherToken::Skip) => {
                    clauses.push(self.parse_skip_clause()?);
                }
                Some(CypherToken::Unwind) => {
                    clauses.push(self.parse_unwind_clause()?);
                }
                Some(CypherToken::Union) => {
                    clauses.push(self.parse_union_clause()?);
                }
                Some(CypherToken::Intersect) => {
                    clauses.push(self.parse_intersect_clause()?);
                }
                Some(CypherToken::Except) => {
                    clauses.push(self.parse_except_clause()?);
                }
                Some(CypherToken::Create) => {
                    clauses.push(self.parse_create_clause()?);
                }
                Some(CypherToken::Set) => {
                    clauses.push(self.parse_set_clause()?);
                }
                Some(CypherToken::Delete) | Some(CypherToken::Detach) => {
                    clauses.push(self.parse_delete_clause()?);
                }
                Some(CypherToken::Remove) => {
                    clauses.push(self.parse_remove_clause()?);
                }
                Some(CypherToken::Merge) => {
                    clauses.push(self.parse_merge_clause()?);
                }
                Some(CypherToken::Call) => {
                    clauses.push(self.parse_call_clause()?);
                }
                Some(CypherToken::Identifier(s)) if s.eq_ignore_ascii_case("FORMAT") => {
                    if end_at_rbrace {
                        return Err(
                            "FORMAT is not allowed inside a CALL { } subquery body".to_string()
                        );
                    }
                    // FORMAT CSV — must be last clause
                    self.advance(); // consume FORMAT
                    match self.peek() {
                        Some(CypherToken::Identifier(fmt)) if fmt.eq_ignore_ascii_case("CSV") => {
                            self.advance(); // consume CSV
                            return Ok((clauses, OutputFormat::Csv));
                        }
                        other => {
                            return Err(format!(
                                "Expected format name after FORMAT (supported: CSV), got {:?}",
                                other
                            ));
                        }
                    }
                }
                Some(t) => {
                    return Err(format!("Unexpected token at start of clause: {:?}", t));
                }
                None => break,
            }
        }

        Ok((clauses, OutputFormat::Default))
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Parse a Cypher query string into a CypherQuery AST.
///
/// On error, enriches the bare token-level message with a source
/// position — `line N col M` plus an excerpt of the source with a
/// caret pointing at the failing position. 0.9.0 §1 / Cluster 3
/// baseline UX: users distinguish "you typo'd" from "feature not
/// yet implemented" by reading the error, not by re-running with
/// `print()`s.
///
/// Position is **byte-precise** — the tokenizer attaches a char
/// offset to every token, the parser threads them through, and
/// `format_parse_error` walks `input.chars()` to convert to
/// (line, col).
/// Parse Cypher source into a typed AST.
///
/// Phase A.2 / C2 — returns [`KgError`] with structured `line` and
/// `col` fields (when the parser knows them) instead of an opaque
/// `Result<_, String>` whose message embedded the position. The
/// position survives the PyO3 boundary and reaches Python consumers
/// via `kglite.CypherSyntaxError.args[0]` (still in the message for
/// human display) and — eventually — as dedicated `.line` / `.col`
/// attributes once PyO3 lands a clean per-exception attribute API.
///
/// The internal tokenizer/parser still produce `Result<_, String>`
/// for ergonomic `?` chains inside the parsing code — only the
/// outer boundary is typed.
pub fn parse_cypher(input: &str) -> Result<CypherQuery, KgError> {
    let positioned =
        super::tokenizer::tokenize_cypher_with_positions(input).map_err(|tokenizer_err| {
            // Tokenizer errors don't carry a position the way parser
            // errors do — they happen during char-stream scanning,
            // before token positions are computed. Surface the
            // message without line/col.
            KgError::CypherSyntax {
                message: tokenizer_err,
                line: None,
                col: None,
            }
        })?;
    let (tokens, positions): (Vec<_>, Vec<_>) = positioned.into_iter().unzip();
    let mut parser = CypherParser::new(tokens);
    match parser.parse_query() {
        Ok(q) => Ok(q),
        Err(e) => {
            // Failing char offset = position of token at parser.pos,
            // or end-of-input if the parser ran past the end.
            let char_offset = positions
                .get(parser.pos)
                .copied()
                .unwrap_or_else(|| input.chars().count());
            let (line, col) = char_offset_to_line_col(input, char_offset);
            // Keep the human-readable excerpt formatting in the
            // message — caret marker, source line — so error output
            // is still informative when only the message is shown.
            // The (line, col) struct fields enable programmatic
            // access for the agent surface.
            let message = format_parse_error_message(input, &e, line, col);
            Err(KgError::CypherSyntax {
                message,
                line: Some(line),
                col: Some(col),
            })
        }
    }
}

/// Convert a char offset (index into `input.chars().collect()`)
/// to a 1-based (line, col) pair by walking the input. Used on
/// the error path, so iteration cost is fine.
fn char_offset_to_line_col(input: &str, target_char: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut col = 1usize;
    for (idx, ch) in input.chars().enumerate() {
        if idx == target_char {
            return (line, col);
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Recognize a small set of "feature not yet implemented" sequences
/// and rewrite the parser error into an intent-level message.
/// Conservative: only reframes when we're confident the original
/// query targeted an unimplemented feature, otherwise returns None.
///
/// Currently a stub — no stable not-yet-implemented features to
/// detect (the named candidates — NULLS, datetime-accessor,
/// variable-length paths — all parse without error today). New §X
/// work plugs in detection here as features land or ship as
/// `not-yet-implemented`.
fn intent_level_rewrite(_input: &str, _err: &str) -> Option<String> {
    None
}

/// Build the human-readable parse-error message body. The (line, col)
/// is included in the message text *and* carried as struct fields on
/// `KgError::CypherSyntax`; the duplication is intentional so the
/// raw message printed by `Display` is still self-contained.
fn format_parse_error_message(input: &str, err: &str, line: usize, col: usize) -> String {
    let intent = intent_level_rewrite(input, err);

    // Build a single-line excerpt of the offending line + a caret
    // marker. Avoids dumping the whole multi-line query.
    let lines: Vec<&str> = input.lines().collect();
    let excerpt = if line >= 1 && line <= lines.len() {
        let src_line = lines[line - 1];
        let caret_col = col.saturating_sub(1).min(src_line.len());
        let caret = format!("{:width$}^", "", width = caret_col);
        format!("\n   {}\n   {}", src_line, caret)
    } else {
        String::new()
    };

    let body = intent.as_deref().unwrap_or(err);
    format!("{}{}", body, excerpt)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_match_return() {
        let query = parse_cypher("MATCH (n:Person) RETURN n").unwrap();
        assert_eq!(query.clauses.len(), 2);
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::Return(_)));
    }

    #[test]
    fn test_match_where_return() {
        let query =
            parse_cypher("MATCH (n:Person) WHERE n.age > 30 RETURN n.name AS name").unwrap();
        assert_eq!(query.clauses.len(), 3);
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::Where(_)));
        assert!(matches!(&query.clauses[2], Clause::Return(_)));

        // Check WHERE predicate
        if let Clause::Where(w) = &query.clauses[1] {
            if let Predicate::Comparison {
                left,
                operator,
                right,
            } = &w.predicate
            {
                assert!(
                    matches!(left, Expression::PropertyAccess { variable, property }
                    if variable == "n" && property == "age")
                );
                assert_eq!(*operator, ComparisonOp::GreaterThan);
                assert!(matches!(right, Expression::Literal(Value::Int64(30))));
            } else {
                panic!("Expected comparison predicate");
            }
        } else {
            panic!("Expected WHERE clause");
        }

        // Check RETURN alias
        if let Clause::Return(r) = &query.clauses[2] {
            assert_eq!(r.items.len(), 1);
            assert_eq!(r.items[0].alias, Some("name".to_string()));
        }
    }

    #[test]
    fn test_where_and_or() {
        let query = parse_cypher(
            "MATCH (n:Person) WHERE n.age > 18 AND n.city = 'Oslo' OR n.vip = true RETURN n",
        )
        .unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            // Should be: (age > 18 AND city = 'Oslo') OR vip = true
            assert!(matches!(&w.predicate, Predicate::Or(_, _)));
        }
    }

    #[test]
    fn test_where_not() {
        let query = parse_cypher("MATCH (n:Person) WHERE NOT n.active = false RETURN n").unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            assert!(matches!(&w.predicate, Predicate::Not(_)));
        }
    }

    #[test]
    fn test_where_is_null() {
        let query = parse_cypher("MATCH (n:Person) WHERE n.email IS NULL RETURN n").unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            assert!(matches!(&w.predicate, Predicate::IsNull(_)));
        }
    }

    #[test]
    fn test_where_is_not_null() {
        let query = parse_cypher("MATCH (n:Person) WHERE n.email IS NOT NULL RETURN n").unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            assert!(matches!(&w.predicate, Predicate::IsNotNull(_)));
        }
    }

    #[test]
    fn test_where_in_list() {
        let query = parse_cypher(
            "MATCH (n:Person) WHERE n.city IN ['Oslo', 'Bergen', 'Trondheim'] RETURN n",
        )
        .unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            if let Predicate::In { expr: _, list } = &w.predicate {
                assert_eq!(list.len(), 3);
            } else {
                panic!("Expected IN predicate");
            }
        }
    }

    #[test]
    fn test_return_multiple_items() {
        let query =
            parse_cypher("MATCH (n:Person) RETURN n.name AS name, n.age AS age, n.city").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            assert_eq!(r.items.len(), 3);
            assert_eq!(r.items[0].alias, Some("name".to_string()));
            assert_eq!(r.items[1].alias, Some("age".to_string()));
            assert_eq!(r.items[2].alias, None);
        }
    }

    #[test]
    fn test_return_distinct() {
        let query = parse_cypher("MATCH (n:Person) RETURN DISTINCT n.city").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            assert!(r.distinct);
        }
    }

    #[test]
    fn test_return_function_call() {
        let query = parse_cypher("MATCH (n:Person) RETURN count(n) AS total").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            if let Expression::FunctionCall {
                name,
                args,
                distinct,
            } = &r.items[0].expression
            {
                assert_eq!(name, "count");
                assert_eq!(args.len(), 1);
                assert!(!distinct);
            } else {
                panic!("Expected function call");
            }
        }
    }

    #[test]
    fn test_return_count_star() {
        let query = parse_cypher("MATCH (n:Person) RETURN count(*) AS total").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            if let Expression::FunctionCall { args, .. } = &r.items[0].expression {
                assert!(matches!(&args[0], Expression::Star));
            }
        }
    }

    #[test]
    fn test_return_count_distinct() {
        let query =
            parse_cypher("MATCH (n:Person) RETURN count(DISTINCT n.city) AS cities").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            if let Expression::FunctionCall { distinct, .. } = &r.items[0].expression {
                assert!(distinct);
            }
        }
    }

    #[test]
    fn test_order_by_limit_skip() {
        let query =
            parse_cypher("MATCH (n:Person) RETURN n.name ORDER BY n.age DESC SKIP 5 LIMIT 10")
                .unwrap();

        assert!(matches!(&query.clauses[2], Clause::OrderBy(_)));
        assert!(matches!(&query.clauses[3], Clause::Skip(_)));
        assert!(matches!(&query.clauses[4], Clause::Limit(_)));

        if let Clause::OrderBy(o) = &query.clauses[2] {
            assert_eq!(o.items.len(), 1);
            assert!(!o.items[0].ascending);
        }
    }

    #[test]
    fn test_with_clause() {
        let query = parse_cypher(
            "MATCH (n:Person) WITH n.city AS city, count(n) AS cnt WHERE cnt > 5 RETURN city, cnt",
        )
        .unwrap();

        assert!(matches!(&query.clauses[1], Clause::With(_)));
        if let Clause::With(w) = &query.clauses[1] {
            assert_eq!(w.items.len(), 2);
            assert!(w.where_clause.is_some());
        }
    }

    #[test]
    fn test_optional_match() {
        let query =
            parse_cypher("MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(f:Person) RETURN n, f")
                .unwrap();

        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::OptionalMatch(_)));
        assert!(matches!(&query.clauses[2], Clause::Return(_)));
    }

    #[test]
    fn test_match_with_edge_pattern() {
        let query =
            parse_cypher("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name").unwrap();

        if let Clause::Match(m) = &query.clauses[0] {
            assert_eq!(m.patterns.len(), 1);
            assert_eq!(m.patterns[0].elements.len(), 3); // node, edge, node
        }
    }

    #[test]
    fn test_match_with_var_length() {
        let query = parse_cypher("MATCH (a:Person)-[:KNOWS*1..3]->(b:Person) RETURN a, b").unwrap();

        assert!(matches!(&query.clauses[0], Clause::Match(_)));
    }

    #[test]
    fn test_multiple_match_patterns() {
        let query = parse_cypher("MATCH (a:Person), (b:Company) RETURN a, b").unwrap();

        if let Clause::Match(m) = &query.clauses[0] {
            assert_eq!(m.patterns.len(), 2);
        }
    }

    #[test]
    fn test_case_insensitive() {
        let query = parse_cypher("match (n:Person) where n.age > 30 return n").unwrap();
        assert_eq!(query.clauses.len(), 3);
    }

    #[test]
    fn test_arithmetic_in_return() {
        let query =
            parse_cypher("MATCH (n:Product) RETURN n.price * 1.1 AS price_with_tax").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            assert!(matches!(&r.items[0].expression, Expression::Multiply(_, _)));
        }
    }

    #[test]
    fn test_where_contains() {
        let query = parse_cypher("MATCH (n:Person) WHERE n.name CONTAINS 'son' RETURN n").unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            assert!(matches!(&w.predicate, Predicate::Contains { .. }));
        }
    }

    #[test]
    fn test_unwind() {
        let query = parse_cypher("UNWIND [1, 2, 3] AS x RETURN x").unwrap();

        assert!(matches!(&query.clauses[0], Clause::Unwind(_)));
        if let Clause::Unwind(u) = &query.clauses[0] {
            assert_eq!(u.alias, "x");
        }
    }

    #[test]
    fn test_case_generic_form() {
        let query = parse_cypher(
            "MATCH (n:Person) RETURN CASE WHEN n.age > 18 THEN 'adult' ELSE 'minor' END AS category",
        )
        .unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            assert!(
                matches!(&r.items[0].expression, Expression::Case { operand, .. } if operand.is_none())
            );
            assert_eq!(r.items[0].alias, Some("category".to_string()));
        } else {
            panic!("Expected RETURN clause");
        }
    }

    #[test]
    fn test_case_simple_form() {
        let query = parse_cypher(
            "MATCH (n:Person) RETURN CASE n.city WHEN 'Oslo' THEN 'capital' WHEN 'Bergen' THEN 'west' ELSE 'other' END",
        )
        .unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            if let Expression::Case {
                operand,
                when_clauses,
                else_expr,
            } = &r.items[0].expression
            {
                assert!(operand.is_some());
                assert_eq!(when_clauses.len(), 2);
                assert!(else_expr.is_some());
            } else {
                panic!("Expected CASE expression");
            }
        }
    }

    #[test]
    fn test_case_no_else() {
        let query =
            parse_cypher("MATCH (n:Person) RETURN CASE WHEN n.age > 18 THEN 'adult' END").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            if let Expression::Case { else_expr, .. } = &r.items[0].expression {
                assert!(else_expr.is_none());
            } else {
                panic!("Expected CASE expression");
            }
        }
    }

    #[test]
    fn test_parameter_in_expression() {
        let query = parse_cypher("MATCH (n:Person) WHERE n.age > $min_age RETURN n.name").unwrap();

        if let Clause::Where(w) = &query.clauses[1] {
            if let Predicate::Comparison { right, .. } = &w.predicate {
                assert!(matches!(right, Expression::Parameter(name) if name == "min_age"));
            } else {
                panic!("Expected comparison predicate");
            }
        }
    }

    #[test]
    fn test_parameter_in_return() {
        let query = parse_cypher("MATCH (n:Person) RETURN n.name, $label AS label").unwrap();

        if let Clause::Return(r) = &query.clauses[1] {
            assert!(
                matches!(&r.items[1].expression, Expression::Parameter(name) if name == "label")
            );
        }
    }

    // ========================================================================
    // CREATE Clause
    // ========================================================================

    #[test]
    fn test_parse_create_node() {
        let query = parse_cypher("CREATE (n:Person {name: 'Alice', age: 30})").unwrap();
        assert_eq!(query.clauses.len(), 1);

        if let Clause::Create(c) = &query.clauses[0] {
            assert_eq!(c.patterns.len(), 1);
            assert_eq!(c.patterns[0].elements.len(), 1);
            if let CreateElement::Node(np) = &c.patterns[0].elements[0] {
                assert_eq!(np.variable, Some("n".to_string()));
                assert_eq!(np.label, Some("Person".to_string()));
                assert_eq!(np.properties.len(), 2);
                assert_eq!(np.properties[0].0, "name");
                assert_eq!(np.properties[1].0, "age");
            } else {
                panic!("Expected node element");
            }
        } else {
            panic!("Expected CREATE clause");
        }
    }

    #[test]
    fn test_parse_create_edge() {
        let query = parse_cypher("MATCH (a:Person), (b:Person) CREATE (a)-[:KNOWS]->(b)").unwrap();
        assert_eq!(query.clauses.len(), 2);
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::Create(_)));

        if let Clause::Create(c) = &query.clauses[1] {
            assert_eq!(c.patterns[0].elements.len(), 3); // node, edge, node
            if let CreateElement::Edge(ep) = &c.patterns[0].elements[1] {
                assert_eq!(ep.connection_type, "KNOWS");
                assert_eq!(ep.direction, CreateEdgeDirection::Outgoing);
            } else {
                panic!("Expected edge element");
            }
        }
    }

    #[test]
    fn test_parse_create_path() {
        let query =
            parse_cypher("CREATE (a:Person {name: 'A'})-[:KNOWS]->(b:Person {name: 'B'})").unwrap();

        if let Clause::Create(c) = &query.clauses[0] {
            assert_eq!(c.patterns[0].elements.len(), 3);
            assert!(matches!(&c.patterns[0].elements[0], CreateElement::Node(_)));
            assert!(matches!(&c.patterns[0].elements[1], CreateElement::Edge(_)));
            assert!(matches!(&c.patterns[0].elements[2], CreateElement::Node(_)));
        }
    }

    #[test]
    fn test_parse_create_with_params() {
        let query = parse_cypher("CREATE (n:Person {name: $name, age: $age})").unwrap();

        if let Clause::Create(c) = &query.clauses[0] {
            if let CreateElement::Node(np) = &c.patterns[0].elements[0] {
                assert!(matches!(&np.properties[0].1, Expression::Parameter(n) if n == "name"));
                assert!(matches!(&np.properties[1].1, Expression::Parameter(n) if n == "age"));
            }
        }
    }

    #[test]
    fn test_parse_create_incoming_edge() {
        let query =
            parse_cypher("MATCH (a:Person), (b:Person) CREATE (a)<-[:FOLLOWS]-(b)").unwrap();

        if let Clause::Create(c) = &query.clauses[1] {
            if let CreateElement::Edge(ep) = &c.patterns[0].elements[1] {
                assert_eq!(ep.connection_type, "FOLLOWS");
                assert_eq!(ep.direction, CreateEdgeDirection::Incoming);
            }
        }
    }

    // ========================================================================
    // SET Clause
    // ========================================================================

    #[test]
    fn test_parse_set_property() {
        let query = parse_cypher("MATCH (n:Person) SET n.age = 31").unwrap();
        assert_eq!(query.clauses.len(), 2);
        assert!(matches!(&query.clauses[1], Clause::Set(_)));

        if let Clause::Set(s) = &query.clauses[1] {
            assert_eq!(s.items.len(), 1);
            if let SetItem::Property {
                variable,
                property,
                expression,
            } = &s.items[0]
            {
                assert_eq!(variable, "n");
                assert_eq!(property, "age");
                assert!(matches!(expression, Expression::Literal(Value::Int64(31))));
            }
        }
    }

    #[test]
    fn test_parse_set_multiple() {
        let query = parse_cypher("MATCH (n:Person) SET n.age = 31, n.city = 'Bergen'").unwrap();

        if let Clause::Set(s) = &query.clauses[1] {
            assert_eq!(s.items.len(), 2);
            if let SetItem::Property { property, .. } = &s.items[0] {
                assert_eq!(property, "age");
            }
            if let SetItem::Property { property, .. } = &s.items[1] {
                assert_eq!(property, "city");
            }
        }
    }

    #[test]
    fn test_parse_set_expression() {
        let query = parse_cypher("MATCH (n:Person) SET n.salary = n.salary * 1.1").unwrap();

        if let Clause::Set(s) = &query.clauses[1] {
            if let SetItem::Property { expression, .. } = &s.items[0] {
                assert!(matches!(expression, Expression::Multiply(_, _)));
            }
        }
    }

    #[test]
    fn test_parse_match_create_set_return() {
        let query = parse_cypher(
            "MATCH (a:Person) CREATE (a)-[:RATED]->(r:Review {text: 'Great'}) SET a.reviews = a.reviews + 1 RETURN a, r",
        ).unwrap();

        assert_eq!(query.clauses.len(), 4);
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        assert!(matches!(&query.clauses[1], Clause::Create(_)));
        assert!(matches!(&query.clauses[2], Clause::Set(_)));
        assert!(matches!(&query.clauses[3], Clause::Return(_)));
    }

    // ========================================================================
    // DELETE Clause
    // ========================================================================

    #[test]
    fn test_parse_delete() {
        let query = parse_cypher("MATCH (n:Person) DELETE n").unwrap();
        assert_eq!(query.clauses.len(), 2);
        if let Clause::Delete(d) = &query.clauses[1] {
            assert!(!d.detach);
            assert_eq!(d.expressions.len(), 1);
            assert!(matches!(&d.expressions[0], Expression::Variable(v) if v == "n"));
        } else {
            panic!("Expected DELETE clause");
        }
    }

    #[test]
    fn test_parse_detach_delete() {
        let query = parse_cypher("MATCH (n:Person) DETACH DELETE n").unwrap();
        if let Clause::Delete(d) = &query.clauses[1] {
            assert!(d.detach);
            assert_eq!(d.expressions.len(), 1);
        } else {
            panic!("Expected DELETE clause");
        }
    }

    #[test]
    fn test_parse_delete_multiple() {
        let query = parse_cypher("MATCH (a)-[r]->(b) DELETE a, r, b").unwrap();
        if let Clause::Delete(d) = &query.clauses[1] {
            assert_eq!(d.expressions.len(), 3);
        }
    }

    // ========================================================================
    // REMOVE Clause
    // ========================================================================

    #[test]
    fn test_parse_remove_property() {
        let query = parse_cypher("MATCH (n:Person) REMOVE n.age").unwrap();
        assert!(matches!(&query.clauses[1], Clause::Remove(_)));
        if let Clause::Remove(r) = &query.clauses[1] {
            assert_eq!(r.items.len(), 1);
            if let RemoveItem::Property { variable, property } = &r.items[0] {
                assert_eq!(variable, "n");
                assert_eq!(property, "age");
            } else {
                panic!("Expected property removal");
            }
        }
    }

    #[test]
    fn test_parse_remove_multiple() {
        let query = parse_cypher("MATCH (n:Person) REMOVE n.age, n.city").unwrap();
        if let Clause::Remove(r) = &query.clauses[1] {
            assert_eq!(r.items.len(), 2);
        }
    }

    #[test]
    fn test_parse_remove_label() {
        let query = parse_cypher("MATCH (n:Person) REMOVE n:Temporary").unwrap();
        if let Clause::Remove(r) = &query.clauses[1] {
            assert!(
                matches!(&r.items[0], RemoveItem::Label { variable, label } if variable == "n" && label == "Temporary")
            );
        }
    }

    // ========================================================================
    // MERGE Clause
    // ========================================================================

    #[test]
    fn test_parse_merge_node() {
        let query = parse_cypher("MERGE (n:Person {name: 'Alice'})").unwrap();
        assert_eq!(query.clauses.len(), 1);
        assert!(matches!(&query.clauses[0], Clause::Merge(_)));
        if let Clause::Merge(m) = &query.clauses[0] {
            assert_eq!(m.pattern.elements.len(), 1);
            assert!(m.on_create.is_none());
            assert!(m.on_match.is_none());
        }
    }

    #[test]
    fn test_parse_merge_on_create() {
        let query =
            parse_cypher("MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.age = 30").unwrap();
        if let Clause::Merge(m) = &query.clauses[0] {
            assert!(m.on_create.is_some());
            assert!(m.on_match.is_none());
            assert_eq!(m.on_create.as_ref().unwrap().len(), 1);
        }
    }

    #[test]
    fn test_parse_merge_on_match() {
        let query =
            parse_cypher("MERGE (n:Person {name: 'Alice'}) ON MATCH SET n.visits = 1").unwrap();
        if let Clause::Merge(m) = &query.clauses[0] {
            assert!(m.on_create.is_none());
            assert!(m.on_match.is_some());
        }
    }

    #[test]
    fn test_parse_merge_both() {
        let query = parse_cypher(
            "MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.age = 30 ON MATCH SET n.visits = 1",
        )
        .unwrap();
        if let Clause::Merge(m) = &query.clauses[0] {
            assert!(m.on_create.is_some());
            assert!(m.on_match.is_some());
        }
    }

    #[test]
    fn test_parse_merge_relationship() {
        let query = parse_cypher("MATCH (a:Person), (b:Person) MERGE (a)-[r:KNOWS]->(b)").unwrap();
        assert_eq!(query.clauses.len(), 2);
        if let Clause::Merge(m) = &query.clauses[1] {
            assert_eq!(m.pattern.elements.len(), 3);
        }
    }

    #[test]
    fn test_reserved_word_as_alias() {
        // Keywords should be valid alias names after AS
        for keyword in &[
            "optional", "match", "where", "return", "order", "limit", "type", "set", "all",
            "distinct", "contains", "exists", "null", "true", "false", "in", "is", "not",
        ] {
            let query_str = format!("MATCH (n) RETURN n AS {}", keyword);
            let query = parse_cypher(&query_str)
                .unwrap_or_else(|e| panic!("Failed to parse 'RETURN n AS {}': {}", keyword, e));
            if let Clause::Return(ret) = &query.clauses[1] {
                assert_eq!(
                    ret.items[0].alias.as_deref(),
                    Some(*keyword),
                    "Alias should be '{}' for keyword",
                    keyword
                );
            } else {
                panic!("Expected RETURN clause");
            }
        }
    }

    #[test]
    fn test_reserved_word_as_unwind_alias() {
        let query = parse_cypher("UNWIND [1,2] AS optional").unwrap();
        if let Clause::Unwind(u) = &query.clauses[0] {
            assert_eq!(u.alias, "optional");
        } else {
            panic!("Expected UNWIND clause");
        }
    }

    #[test]
    fn test_reserved_word_as_yield_alias() {
        let query = parse_cypher("CALL pagerank() YIELD node AS optional, score AS limit").unwrap();
        if let Clause::Call(c) = &query.clauses[0] {
            assert_eq!(c.yield_items[0].alias.as_deref(), Some("optional"));
            assert_eq!(c.yield_items[1].alias.as_deref(), Some("limit"));
        } else {
            panic!("Expected CALL clause");
        }
    }

    // ========================================================================
    // CALL { } subqueries (Phase 1: parser + AST node)
    // ========================================================================

    #[test]
    fn test_call_subquery_uncorrelated() {
        let query =
            parse_cypher("CALL { MATCH (n:Person) RETURN count(n) AS c } RETURN c").unwrap();
        assert_eq!(query.clauses.len(), 2);
        if let Clause::CallSubquery { import, body } = &query.clauses[0] {
            assert!(import.is_empty(), "uncorrelated subquery has no imports");
            assert_eq!(body.clauses.len(), 2);
            assert!(matches!(&body.clauses[0], Clause::Match(_)));
            assert!(matches!(&body.clauses[1], Clause::Return(_)));
        } else {
            panic!("Expected CallSubquery, got {:?}", query.clauses[0]);
        }
        assert!(matches!(&query.clauses[1], Clause::Return(_)));
    }

    #[test]
    fn test_call_subquery_correlated_importing_with() {
        let query = parse_cypher(
            "MATCH (p:Person) CALL { WITH p MATCH (p)-[:KNOWS]->(f) RETURN count(f) AS c } \
             RETURN p.name, c",
        )
        .unwrap();
        assert_eq!(query.clauses.len(), 3);
        assert!(matches!(&query.clauses[0], Clause::Match(_)));
        if let Clause::CallSubquery { import, body } = &query.clauses[1] {
            assert_eq!(import, &vec!["p".to_string()]);
            // The importing WITH is stripped from the body.
            assert_eq!(body.clauses.len(), 2);
            assert!(matches!(&body.clauses[0], Clause::Match(_)));
            assert!(matches!(&body.clauses[1], Clause::Return(_)));
        } else {
            panic!("Expected CallSubquery, got {:?}", query.clauses[1]);
        }
    }

    #[test]
    fn test_call_subquery_multi_import() {
        let query = parse_cypher(
            "MATCH (p:Person), (q:Person) CALL { WITH p, q MATCH (p)-[:KNOWS]->(q) \
             RETURN count(*) AS c } RETURN c",
        )
        .unwrap();
        if let Clause::CallSubquery { import, .. } = &query.clauses[1] {
            assert_eq!(import, &vec!["p".to_string(), "q".to_string()]);
        } else {
            panic!("Expected CallSubquery");
        }
    }

    #[test]
    fn test_call_subquery_nested_braces() {
        // Nested CALL {} plus a map literal containing a '}' must not close early.
        let query = parse_cypher(
            "CALL { CALL { MATCH (n) RETURN n LIMIT 1 } MATCH (m {tag: 'x}y'}) \
             RETURN m AS r, {a: 1, b: 2} AS meta } RETURN r, meta",
        )
        .unwrap();
        assert_eq!(query.clauses.len(), 2);
        if let Clause::CallSubquery { import, body } = &query.clauses[0] {
            assert!(import.is_empty());
            // body = [nested CallSubquery, MATCH, RETURN]
            assert_eq!(body.clauses.len(), 3);
            assert!(matches!(&body.clauses[0], Clause::CallSubquery { .. }));
            assert!(matches!(&body.clauses[1], Clause::Match(_)));
            assert!(matches!(&body.clauses[2], Clause::Return(_)));
        } else {
            panic!("Expected outer CallSubquery, got {:?}", query.clauses[0]);
        }
    }

    #[test]
    fn test_call_subquery_map_literal_not_closing() {
        // A RETURN'd map literal at the end of the body must not be mistaken
        // for the subquery's closing brace.
        let query = parse_cypher("CALL { MATCH (n) RETURN {a: n.x} AS m } RETURN m").unwrap();
        assert_eq!(query.clauses.len(), 2);
        assert!(matches!(&query.clauses[0], Clause::CallSubquery { .. }));
        assert!(matches!(&query.clauses[1], Clause::Return(_)));
    }

    #[test]
    fn test_call_subquery_missing_closing_brace() {
        let err = parse_cypher("CALL { MATCH (n:Person) RETURN n").unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains('}') || msg.to_lowercase().contains("close"),
            "expected a missing-brace error, got: {msg}"
        );
    }

    #[test]
    fn test_call_subquery_importing_with_alias_rejected() {
        // `WITH p AS x` in the importing position is illegal.
        let err = parse_cypher(
            "MATCH (p:Person) CALL { WITH p AS x MATCH (x) RETURN count(x) AS c } RETURN c",
        )
        .unwrap_err();
        let msg = format!("{}", err).to_lowercase();
        assert!(
            msg.contains("importing with"),
            "expected importing-WITH violation, got: {msg}"
        );
    }

    #[test]
    fn test_call_subquery_importing_with_projection_rejected() {
        // `WITH p.name` (a projection) in the importing position is illegal.
        let err = parse_cypher("MATCH (p:Person) CALL { WITH p.name MATCH (n) RETURN n } RETURN n")
            .unwrap_err();
        let msg = format!("{}", err).to_lowercase();
        assert!(
            msg.contains("importing with"),
            "expected importing-WITH violation, got: {msg}"
        );
    }

    #[test]
    fn test_call_procedure_still_parses() {
        // The existing CALL procedure form must be unaffected by the
        // subquery branch.
        let query = parse_cypher("CALL pagerank() YIELD node, score RETURN node, score").unwrap();
        assert!(matches!(&query.clauses[0], Clause::Call(_)));
        if let Clause::Call(c) = &query.clauses[0] {
            assert_eq!(c.procedure_name, "pagerank");
            assert_eq!(c.yield_items.len(), 2);
        }
    }
}
