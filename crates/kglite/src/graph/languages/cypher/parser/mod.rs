//! Cypher parser — delegates MATCH patterns to
//! `crate::graph::core::pattern_matching::parse_pattern`.
//!
//! Split (Phase 9):
//! - [`match_pattern`] — MATCH / OPTIONAL MATCH, pattern extraction, EXISTS
//! - [`predicate`] — WHERE predicate chain (AND / OR / XOR / NOT / comparisons)
//! - [`expression`] — expressions (arithmetic, function calls, CASE, list ops)
//! - [`clauses`] — RETURN / WITH / ORDER BY / LIMIT / SKIP / UNWIND / UNION /
//!   CREATE / SET / DELETE / REMOVE / MERGE / CALL
//! - [`schema_ddl`] — CREATE/DROP/SHOW INDEX and CONSTRAINT (Neo4j 5 DDL)
//! - [`load_csv`] — LOAD CSV (leading-position external row source)
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
pub mod load_csv;
pub mod match_pattern;
pub mod predicate;
pub mod schema_ddl;

/// Tokenizes and parses Cypher query strings into a `CypherQuery` AST.
///
/// Handles the full Cypher clause set: MATCH, WHERE, RETURN, WITH,
/// ORDER BY, LIMIT, SKIP, CREATE, SET, DELETE, MERGE, REMOVE, UNWIND, UNION.
/// Uses a token-based recursive descent approach.
pub struct CypherParser {
    tokens: Vec<CypherToken>,
    pos: usize,
    /// Nesting depth of the expression/predicate AST currently under
    /// construction. Charged by [`Self::descend`] for *recursive* nesting
    /// (parens/lists/`NOT`) and by [`Self::deepen`] for the *iteratively*
    /// built left-associative operator chains, so pathologically nested
    /// input returns a parse error instead of overflowing the stack and
    /// aborting the process.
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

/// Maximum expression/predicate AST nesting depth accepted by the parser.
///
/// The recursive-descent expression parser, the planner's expression walkers,
/// the executor's `evaluate_expression` and the AST's `Drop` all recurse once
/// per nesting level, so this budget bounds the *depth* every one of them
/// walks. 512 levels is far beyond any legitimate query.
///
/// It does not, however, bound their *stack use* equally. Only
/// [`CypherParser::descend`] grows the stack on demand (via `stacker`); the
/// planner, executor and `Drop` walkers recurse on whatever stack their
/// thread already has. That has now been measured
/// (`super::stack_probe`, macOS/aarch64, worst shape: an `OR`/`AND`/`NOT`
/// tree walked by the executor):
///
/// | stage | debug | release |
/// |---|---|---|
/// | executor `evaluate_predicate`/`evaluate_expression` | 7,248 B/level | 1,056 B/level |
/// | planner walkers (WHERE chains) | 2,160 B/level | 480 B/level |
/// | recursive `Drop` | 112 B/level | 64 B/level |
///
/// At this budget the deepest accepted query therefore peaks at ~3.7 MiB in
/// debug and ~0.54 MiB in release, against the 8 MiB every frontend runs on
/// ([`crate::graph::session::QUERY_THREAD_STACK_SIZE`], and the main-thread
/// default for the CLI and the Python wheel). So the stages downstream of the
/// parser do not need their own `stacker` guard — the budget alone keeps them
/// inside the stack they are given, and
/// `stack_probe::budget_ceiling_query_fits_the_query_thread_stack` holds that
/// true on every platform CI runs.
///
/// The corollary is that this budget is load-bearing, not decorative: raising
/// it scales the peak linearly, and the release peak would pass 1 MiB — the
/// smallest in-process stack in play — somewhere around 1,000 levels. Raising
/// the budget means adding a guard first, and a guard on the per-row
/// `evaluate_expression` measured **+14%** on the in-memory expression path.
/// Growing the stack once at executor entry measured free, and is the shape
/// to reach for if the budget ever has to move.
///
/// The budget bounds **AST depth, not parser call depth**, and those differ:
/// the left-associative operator chains (`OR`/`XOR`/`AND`, `+`/`-`/`||`,
/// `*`/`/`/`%`, subscripting, `n:A:B` label chains) are parsed *iteratively*,
/// so the parser itself stays shallow while each iteration adds one level to
/// the tree it returns. Those levels are charged via [`CypherParser::deepen`]
/// inside a [`CypherParser::chain`] scope; recursive nesting is charged by
/// [`CypherParser::descend`]. Charging both is what makes the budget an
/// actual bound on what the downstream walkers recurse through — the
/// walkers themselves have neither a depth guard nor stack growth, and they
/// run on server worker threads whose stacks are far smaller than the main
/// thread's.
pub(super) const MAX_EXPRESSION_DEPTH: usize = 512;

/// Remaining-stack threshold below which [`CypherParser::descend`] allocates
/// a fresh segment, and the size of that segment. The red zone must cover the
/// deepest frame chain one nesting level can add *while parsing* (~10 frames
/// in debug); the plan/execute/drop walkers are not covered, since
/// `maybe_grow` is only called from the parser.
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
        self.deepen()?;
        let result = stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || f(self));
        self.depth -= 1;
        result
    }

    /// Charge one level of AST nesting against [`MAX_EXPRESSION_DEPTH`],
    /// failing with a clean parse error once the budget is exhausted.
    ///
    /// Call this — inside a [`Self::chain`] scope — once per iteration of an
    /// iteratively-parsed operator chain, because each iteration wraps the
    /// accumulated operand in one more AST node. Unlike [`Self::descend`]
    /// the charge is *not* released when the current call returns; `chain`
    /// releases the whole run at once.
    pub(super) fn deepen(&mut self) -> Result<(), String> {
        if self.depth >= MAX_EXPRESSION_DEPTH {
            // Name the rewrite. The overwhelmingly common way to reach this
            // limit is generated code — a filter/facet builder emitting one
            // `OR` term per selected value — and each term costs a nesting
            // level while the equivalent `IN [...]` costs one level no matter
            // how many values it holds. "Simplify the query" alone leaves the
            // author guessing at a limit they did not know existed.
            return Err(format!(
                "Expression nesting exceeds {} levels; simplify the query. \
                 Long chains of OR'd equality tests are the usual cause and \
                 nest one level per term — rewrite `x = a OR x = b OR ...` as \
                 `x IN [a, b, ...]`, which nests one level however many \
                 values the list holds.",
                MAX_EXPRESSION_DEPTH
            ));
        }
        self.depth += 1;
        Ok(())
    }

    /// Run `f` as an iteratively-built operator chain, releasing every level
    /// it charged via [`Self::deepen`] once the chain is complete.
    ///
    /// Releasing on exit is what makes the budget track AST *depth* rather
    /// than total node count: two sibling chains each 400 levels deep form a
    /// tree only 401 levels deep, so the second must not inherit the first's
    /// charges. A chain nested inside another chain's operand is still parsed
    /// while the outer chain holds its charges, so genuine nesting keeps
    /// accumulating — the bound stays conservative in the safe direction.
    pub(super) fn chain<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, String>,
    ) -> Result<T, String> {
        let entry_depth = self.depth;
        let result = f(self);
        self.depth = entry_depth;
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

    // ========================================================================
    // Soft-keyword helpers
    // ========================================================================
    //
    // Several Cypher constructs are spelled with words the tokenizer
    // deliberately does *not* reserve — `INDEX`, `CONSTRAINT`, `FOR`,
    // `OPTIONS` (schema DDL) and `LOAD`, `CSV`, `HEADERS`, `FROM`,
    // `FIELDTERMINATOR` (`LOAD CSV`). Reserving them would break every graph
    // that stores a property or label of the same name, so they arrive as
    // [`CypherToken::Identifier`] and are matched case-insensitively here.
    // Shared by `schema_ddl` and `load_csv`.

    /// The identifier lexeme `offset` tokens ahead, if that token is one.
    pub(super) fn soft_word_at(&self, offset: usize) -> Option<&str> {
        match self.peek_at(offset) {
            Some(CypherToken::Identifier(word)) => Some(word.as_str()),
            _ => None,
        }
    }

    /// True when the next token is the soft keyword `word`.
    pub(super) fn peek_soft_word(&self, word: &str) -> bool {
        self.soft_word_at(0).is_some_and(|w| soft_word_eq(w, word))
    }

    /// Consume the soft keyword `word` if it is next; report whether it was.
    pub(super) fn eat_soft_word(&mut self, word: &str) -> bool {
        if self.peek_soft_word(word) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Consume the soft keyword `word`, or fail naming `context`.
    pub(super) fn expect_soft_word(&mut self, word: &str, context: &str) -> Result<(), String> {
        if self.eat_soft_word(word) {
            Ok(())
        } else {
            Err(format!(
                "Expected {word} in {context}, got {:?}",
                self.peek()
            ))
        }
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
            // `LOAD CSV` is spelled with soft keywords, so it arrives as an
            // identifier and would otherwise be swallowed by the preceding
            // clause's expression/pattern parser — `MATCH (n) LOAD CSV FROM …`
            // died on the `AS` with a pattern-property error instead of the
            // positional rule. Stopping here hands the token back to the
            // clause loop, which reports
            // `CypherParser::misplaced_load_csv_error`.
            Some(CypherToken::Identifier(_)) => self.identifier_opens_load_csv(),
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

        // Schema DDL is a whole statement, not a pipeline stage, so it is
        // recognised here rather than inside the clause loop: one check per
        // query instead of one per clause, and `parse_clause_sequence` keeps
        // its shape. A DDL statement consumes the entire token stream.
        if let Some(clause) = self.try_parse_schema_ddl_statement()? {
            return Ok(CypherQuery {
                clauses: vec![clause],
                explain,
                profile,
                output_format: OutputFormat::Default,
                optimizer_tags: Vec::new(),
            });
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
    /// Parse a leading `LOAD CSV`, or report the positional rule.
    ///
    /// Recognised even when misplaced, so a user who put it after a `MATCH`
    /// gets the rule instead of `Unexpected token at start of clause:
    /// Identifier("LOAD")`.
    fn parse_leading_load_csv(&mut self, first: bool, in_subquery: bool) -> Result<Clause, String> {
        if !first || in_subquery {
            return Err(Self::misplaced_load_csv_error());
        }
        self.parse_load_csv_clause()
    }

    /// Parse the trailing `FORMAT <name>` marker. `FORMAT CSV` is the only
    /// supported spelling, and it is rejected inside a `CALL { }` body.
    fn parse_format_tail(&mut self, in_subquery: bool) -> Result<OutputFormat, String> {
        if in_subquery {
            return Err("FORMAT is not allowed inside a CALL { } subquery body".to_string());
        }
        self.advance(); // consume FORMAT
        match self.peek() {
            Some(CypherToken::Identifier(fmt)) if fmt.eq_ignore_ascii_case("CSV") => {
                self.advance(); // consume CSV
                Ok(OutputFormat::Csv)
            }
            other => Err(format!(
                "Expected format name after FORMAT (supported: CSV), got {:?}",
                other
            )),
        }
    }

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
                // The two soft-keyword clause heads. Both arrive as
                // `Identifier` (neither word is reserved), and both own a
                // positional rule, so each parses in its own method rather
                // than inline — see `parse_leading_load_csv` and
                // `parse_format_tail`.
                Some(CypherToken::Identifier(_)) if self.identifier_opens_load_csv() => {
                    clauses.push(self.parse_leading_load_csv(clauses.is_empty(), end_at_rbrace)?)
                }
                Some(CypherToken::Identifier(s)) if s.eq_ignore_ascii_case("FORMAT") => {
                    return Ok((clauses, self.parse_format_tail(end_at_rbrace)?))
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

/// Case-insensitive comparison of an identifier lexeme against a canonical
/// soft keyword. Shared by the `schema_ddl` and `load_csv` clause parsers.
pub(super) fn soft_word_eq(candidate: &str, canonical: &str) -> bool {
    candidate.eq_ignore_ascii_case(canonical)
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
