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
    /// Current recursion depth of the expression/predicate parser. Guarded by
    /// [`Self::descend`] so pathologically nested input (thousands of nested
    /// parens/lists/`NOT`s) returns a parse error instead of overflowing the
    /// stack and aborting the process.
    depth: usize,
    /// Verbatim source lexeme per keyword-token index (see
    /// [`super::tokenizer::TokenizedCypher::keyword_lexemes`]). Keyword
    /// tokens are unit variants, so a keyword used as a NAME (property key,
    /// label, rel type, alias) recovers its exact source spelling here —
    /// `{order: 1}` stores key `order`, not the canonical `ORDER`. When a
    /// token index is absent the parser falls back to the canonical keyword
    /// spelling.
    keyword_lexemes: std::collections::HashMap<usize, String>,
}

/// Maximum expression-nesting depth accepted by the parser.
///
/// The recursive-descent expression parser, the planner's expression walkers,
/// and the executor's `evaluate_expression` all recurse once per nesting
/// level, so this budget bounds stack use across the whole pipeline (parse →
/// plan → execute → drop). 512 levels is far beyond any legitimate query.
/// Debug-profile frames are several times larger than release frames — deep
/// nesting within this budget can exhaust a default thread stack before the
/// guard fires — so [`CypherParser::descend`] also grows the stack on demand
/// via `stacker`; the budget is the semantic contract, not the overflow
/// protection.
const MAX_EXPRESSION_DEPTH: usize = 512;

/// Remaining-stack threshold below which [`CypherParser::descend`] allocates
/// a fresh segment, and the size of that segment. The red zone must cover the
/// deepest frame chain one nesting level can add across parse/plan/execute
/// walkers (~10 frames in debug).
const STACK_RED_ZONE: usize = 128 * 1024;
const STACK_GROW_SIZE: usize = 4 * 1024 * 1024;

impl CypherParser {
    /// Construct with the tokenizer's verbatim keyword-lexeme table —
    /// the production path (`parse_cypher`). An empty table is valid;
    /// such parsers fall back to canonical keyword spellings in name
    /// position.
    pub fn with_keyword_lexemes(
        tokens: Vec<CypherToken>,
        keyword_lexemes: Vec<(usize, String)>,
    ) -> Self {
        CypherParser {
            tokens,
            pos: 0,
            depth: 0,
            keyword_lexemes: keyword_lexemes.into_iter().collect(),
        }
    }

    /// Verbatim source lexeme of the keyword token at `idx`, when the
    /// parser was built with the tokenizer's lexeme table.
    pub(super) fn keyword_lexeme_at(&self, idx: usize) -> Option<&str> {
        self.keyword_lexemes.get(&idx).map(String::as_str)
    }

    /// Run `f` one expression-nesting level deeper, failing with a clean
    /// parse error once [`MAX_EXPRESSION_DEPTH`] is exceeded. Every
    /// self-recursive entry point of the expression parser (primary
    /// expressions, `NOT` chains, unary minus chains) must route through
    /// this guard.
    pub(super) fn descend<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        if self.depth >= MAX_EXPRESSION_DEPTH {
            return Err(format!(
                "Expression nesting exceeds {} levels; simplify the query",
                MAX_EXPRESSION_DEPTH
            ));
        }
        self.depth += 1;
        let result = stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || f(self));
        self.depth -= 1;
        result
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
    /// Case-preserving: a keyword alias keeps its verbatim source spelling
    /// (`AS Order` names the column `Order`), falling back to the canonical
    /// lowercase word when no lexeme table is present (unit tests).
    pub(super) fn try_consume_alias_name(&mut self) -> Result<String, String> {
        match self.advance().cloned() {
            Some(CypherToken::Identifier(name)) => Ok(name),
            Some(ref token) => token_to_keyword_name(token)
                .map(|canonical| {
                    self.keyword_lexeme_at(self.pos - 1)
                        .map(str::to_string)
                        .unwrap_or(canonical)
                })
                .ok_or_else(|| format!("Expected alias name after AS, got {:?}", token)),
            None => Err("Expected alias name after AS".to_string()),
        }
    }

    /// Consume the next token as a NAME — a node label, relationship type, or
    /// property key. Accepts an identifier verbatim, or a soft-reservable
    /// keyword via `keyword_name_token` (KG-2: `[:CONTAINS]`, `(:CONTAINS)`,
    /// `{contains: 1}`). `context` names the position for the error message,
    /// preserving the original "Expected <X>" wording. Case-preserving: a
    /// keyword name keeps its verbatim source spelling (`{order: 1}` stores
    /// key `order`), falling back to the canonical uppercase word when no
    /// lexeme table is present (unit tests).
    pub(super) fn expect_name(&mut self, context: &str) -> Result<String, String> {
        match self.advance().cloned() {
            Some(CypherToken::Identifier(name)) => Ok(name),
            Some(ref token) => keyword_name_token(token)
                .map(|canonical| {
                    self.keyword_lexeme_at(self.pos - 1)
                        .map(str::to_string)
                        .unwrap_or_else(|| canonical.to_string())
                })
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
            | Some(CypherToken::Foreach)
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
            optimizer_tags: Vec::new(),
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
                Some(CypherToken::Union)
                | Some(CypherToken::Intersect)
                | Some(CypherToken::Except)
                    if end_at_rbrace =>
                {
                    // v1: UNION / INTERSECT / EXCEPT inside a CALL { }
                    // body are deferred (§1.4 / §6 Q2 of the design doc).
                    // Reject here with a precise message — otherwise the
                    // set-op arm parser greedily consumes to EOF and dies
                    // on the closing `}` with a confusing token error.
                    return Err(
                        "UNION / INTERSECT / EXCEPT inside a CALL { } subquery is not supported \
                         in this version"
                            .to_string(),
                    );
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
                Some(CypherToken::Foreach) => {
                    clauses.push(self.parse_foreach_clause()?);
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
/// human display) and as dedicated `.line` / `.col` attributes.
///
/// The internal tokenizer/parser still produce `Result<_, String>`
/// for ergonomic `?` chains inside the parsing code — only the
/// outer boundary is typed.
// KgError deliberately carries structured context; boxing it would change the public result type.
#[allow(clippy::result_large_err)]
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
    let keyword_lexemes = positioned.keyword_lexemes;
    let (tokens, positions): (Vec<_>, Vec<_>) = positioned.tokens.into_iter().unzip();
    let mut parser = CypherParser::with_keyword_lexemes(tokens, keyword_lexemes);
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
#[path = "parser_tests.rs"]
mod parser_tests;
