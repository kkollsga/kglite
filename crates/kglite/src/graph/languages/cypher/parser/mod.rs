//! Cypher parser — delegates MATCH patterns to
//! `crate::graph::core::pattern_matching::parse_pattern`.
//!
//! Submodules:
//! - [`match_pattern`] — MATCH / OPTIONAL MATCH, pattern extraction, EXISTS
//! - [`predicate`] — WHERE predicate chain (AND / OR / XOR / NOT / comparisons)
//! - [`expression`] — expressions (arithmetic, function calls, CASE, list ops)
//! - [`clauses`] — RETURN / WITH / ORDER BY / LIMIT / SKIP / UNWIND / UNION /
//!   CREATE / SET / DELETE / REMOVE / MERGE / CALL
//! - [`schema_ddl`] — CREATE/DROP/SHOW INDEX and CONSTRAINT (Neo4j 5 DDL)
//! - [`load_csv`] — LOAD CSV (leading-position external row source)

use super::ast::*;
use super::tokenizer::{
    describe_token, describe_token_opt, keyword_name_token, reserved_literal_name_token,
    token_to_keyword_name, CypherToken,
};
#[cfg(test)]
use crate::datatypes::values::Value;
use crate::error::KgError;
use crate::graph::core::pattern_matching::ParamLabel;

pub mod clauses;
pub mod expression;
pub mod load_csv;
pub mod match_pattern;
pub mod predicate;
pub mod schema_ddl;

/// Tokenizes and parses Cypher query strings into a `CypherQuery` AST by
/// recursive descent over the token stream.
pub struct CypherParser {
    tokens: Vec<CypherToken>,
    pos: usize,
    /// Nesting depth of the expression/predicate AST currently under
    /// construction, charged against [`MAX_EXPRESSION_DEPTH`] so
    /// pathologically nested input returns a parse error instead of
    /// overflowing the stack and aborting the process. See that constant
    /// for what [`Self::descend`] and [`Self::deepen`] each charge.
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
/// thread already has. Measured (`super::stack_probe`, macOS/aarch64, worst
/// shape: an `OR`/`AND`/`NOT` tree walked by the executor):
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
/// actual bound on what the downstream walkers recurse through.
pub(super) const MAX_EXPRESSION_DEPTH: usize = 512;

/// Remaining-stack threshold below which [`CypherParser::descend`] allocates
/// a fresh segment, and the size of that segment. The red zone must cover the
/// deepest frame chain one nesting level can add *while parsing* (~10 frames
/// in debug); the plan/execute/drop walkers are not covered, since
/// `maybe_grow` is only called from the parser.
const STACK_RED_ZONE: usize = 128 * 1024;
const STACK_GROW_SIZE: usize = 4 * 1024 * 1024;

impl CypherParser {
    /// Construct with the tokenizer's verbatim keyword-lexeme table. An
    /// empty table is valid: such parsers fall back to canonical keyword
    /// spellings in name position.
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
            // The limit is overwhelmingly reached by generated `OR` chains
            // (a filter/facet builder emitting one term — one nesting level
            // — per selected value), so the message names the `IN [...]`
            // rewrite instead of leaving the author to guess at a limit they
            // did not know existed.
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

    pub(super) fn check(&self, token: &CypherToken) -> bool {
        self.peek() == Some(token)
    }

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
                "Expected {word} in {context}, got {}",
                describe_token_opt(self.peek())
            ))
        }
    }

    /// Consume the next token as an alias name (after AS).
    /// Accepts identifiers and reserved keywords (e.g. `AS optional`, `AS type`).
    /// Case-preserving: a keyword alias keeps its verbatim source spelling
    /// (`AS Order` names the column `Order`), falling back to the canonical
    /// lowercase word when no lexeme table is present.
    pub(super) fn try_consume_alias_name(&mut self) -> Result<String, String> {
        match self.advance().cloned() {
            Some(CypherToken::Identifier(name)) => Ok(name),
            Some(ref token) => token_to_keyword_name(token)
                .map(|canonical| {
                    self.keyword_lexeme_at(self.pos - 1)
                        .map(str::to_string)
                        .unwrap_or(canonical)
                })
                .ok_or_else(|| {
                    format!(
                        "Expected alias name after AS, got {}",
                        describe_token(token)
                    )
                }),
            None => Err("Expected alias name after AS".to_string()),
        }
    }

    /// Consume the next token as a NAME — a node label, relationship type, or
    /// property key. Accepts an identifier verbatim, a soft-reservable
    /// keyword via `keyword_name_token` (KG-2: `[:CONTAINS]`, `(:CONTAINS)`,
    /// `{contains: 1}`), or one of the value-literal words via
    /// `reserved_literal_name_token` (`(:TRUE)`, `{null: 1}` — legal schema
    /// names in openCypher 9, where `SchemaName = SymbolicName | ReservedWord`).
    /// `context` names the position for the error message, preserving the
    /// original "Expected <X>" wording. Case-preserving: a keyword name keeps
    /// its verbatim source spelling (`{order: 1}` stores key `order`), falling
    /// back to the canonical uppercase word when no lexeme table is present.
    ///
    /// This is a **name position by construction** — every caller has already
    /// consumed the `:`, `.` or `{` that makes the next token a name — so
    /// accepting TRUE / FALSE / NULL here cannot reach a value position. The
    /// value positions are parsed by `parse_expression` / `parse_value`, which
    /// never route through this function; `{x: true}` stays a boolean.
    pub(super) fn expect_name(&mut self, context: &str) -> Result<String, String> {
        match self.advance().cloned() {
            Some(CypherToken::Identifier(name)) => Ok(name),
            Some(ref token) => keyword_name_token(token)
                .or_else(|| reserved_literal_name_token(token))
                .map(|canonical| {
                    self.keyword_lexeme_at(self.pos - 1)
                        .map(str::to_string)
                        .unwrap_or_else(|| canonical.to_string())
                })
                .ok_or_else(|| format!("Expected {}, got {}", context, describe_with_hint(token))),
            None => Err(format!("Expected {}", context)),
        }
    }

    /// [`Self::expect_name`] for a **label / relationship-type** position,
    /// which additionally accepts a parameter reference (`$label`, `$(label)`).
    ///
    /// Returns the text to park in the string slot — the name itself, or the
    /// `$name` placeholder — plus the parameter name when it *was* a
    /// reference. Callers record that reference alongside the slot, and
    /// [`super::dynamic_labels::resolve`] substitutes it before validation.
    /// See [`ParamLabel`] for why the marker is out of band.
    ///
    /// Name positions reached through the MATCH pattern re-serializer do not
    /// come here: that path hands `$name` to the secondary pattern lexer,
    /// which has the mirror of this function.
    pub(super) fn expect_label_name(
        &mut self,
        context: &str,
    ) -> Result<(String, Option<String>), String> {
        if let Some(CypherToken::Parameter(param)) = self.peek().cloned() {
            self.advance();
            return Ok((ParamLabel::placeholder(&param), Some(param)));
        }
        Ok((self.expect_name(context)?, None))
    }

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
            Some(CypherToken::Optional) => self.peek_at(1) == Some(&CypherToken::Match),
            Some(CypherToken::Order) => self.peek_at(1) == Some(&CypherToken::By),
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

    pub fn parse_query(&mut self) -> Result<CypherQuery, String> {
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

        // A bare `CALL proc()` (no YIELD) is legal only as the entire
        // statement — Neo4j's standalone-CALL rule. Mid-pipeline, YIELD is
        // required: `execute_call` replaces the incoming row set instead of
        // joining with it, so accepting the bare form there would silently
        // drop bound rows.
        if clauses.len() > 1 {
            for clause in &clauses {
                if matches!(&clause, Clause::Call(call) if call.yield_items.is_empty()) {
                    return Err(
                        "CALL requires a YIELD clause when combined with other clauses, \
                         e.g. CALL pagerank() YIELD node, score. A bare CALL is only \
                         valid as the entire statement."
                            .to_string(),
                    );
                }
            }
        }

        Ok(CypherQuery {
            clauses,
            explain,
            profile,
            output_format,
            optimizer_tags: Vec::new(),
        })
    }

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
                "Expected format name after FORMAT (supported: CSV), got {}",
                describe_token_opt(other)
            )),
        }
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
            if end_at_rbrace && self.check(&CypherToken::RBrace) {
                break;
            }

            if self.check(&CypherToken::Semicolon) {
                self.advance();
                continue;
            }

            match self.peek() {
                Some(CypherToken::Match) => {
                    clauses.push(self.parse_match_clause(false)?);
                }
                Some(CypherToken::Optional) => {
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
                    // body are deferred. Reject here with a precise message
                    // — otherwise the
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
                // `Identifier` (neither word is reserved) and each owns a
                // positional rule, so each parses in its own method rather
                // than inline.
                Some(CypherToken::Identifier(_)) if self.identifier_opens_load_csv() => {
                    clauses.push(self.parse_leading_load_csv(clauses.is_empty(), end_at_rbrace)?)
                }
                Some(CypherToken::Identifier(s)) if s.eq_ignore_ascii_case("FORMAT") => {
                    return Ok((clauses, self.parse_format_tail(end_at_rbrace)?))
                }
                Some(t) => {
                    return Err(format!(
                        "Unexpected token at start of clause: {}",
                        describe_token(t)
                    ));
                }
                None => break,
            }
        }

        Ok((clauses, OutputFormat::Default))
    }
}

/// The offending token as the user wrote it, plus — when it is a reserved
/// keyword in a **name** position — the escape hatch it needs.
///
/// Backticks make any word usable as a variable, label, relationship type or
/// property key (`` MATCH (`match`) ``). Without the hint the error states
/// the rule and hides the one-character fix. Non-keyword tokens get no
/// suffix, so the hint never fires on `n.1` or `SET 1 = 2`.
pub(super) fn describe_with_hint(token: &CypherToken) -> String {
    match token_to_keyword_name(token) {
        Some(word) => format!(
            "{} — a reserved keyword; backtick it (`{word}`) to use it as a name",
            describe_token(token)
        ),
        None => describe_token(token),
    }
}

/// [`describe_with_hint`] for a lookahead that may have run off the end.
pub(super) fn describe_with_hint_opt(token: Option<&CypherToken>) -> String {
    match token {
        Some(token) => describe_with_hint(token),
        None => describe_token_opt(None),
    }
}

const BLOCK_COMMENT_UNSUPPORTED: &str =
    "Block comments (/* ... */) are not supported; use a // line comment instead";

pub(super) fn soft_word_eq(candidate: &str, canonical: &str) -> bool {
    candidate.eq_ignore_ascii_case(canonical)
}

/// Parse Cypher source into a typed AST.
///
/// Errors carry a **character-precise** source position: the bare token-level
/// message is enriched with `line N col M` plus an excerpt with a caret at the
/// failing token (the tokenizer attaches a char offset to every token and the
/// parser threads them through), and the same position is repeated as
/// structured `line` / `col` fields on [`KgError`]. Those fields survive the
/// PyO3 boundary and reach Python as `kglite.CypherSyntaxError.line` / `.col`,
/// with the message still in `args[0]` for human display. The internal
/// tokenizer/parser keep returning `Result<_, String>` for ergonomic `?`
/// chains — only this outer boundary is typed.
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
    // Block comments are unimplemented. The tokenizer marks `/*` with a
    // pseudo-token instead of failing, so the rejection can go through the
    // position machinery below — a tokenizer error would carry no line/col
    // and no caret, and the parser would otherwise report the raw `/` as
    // `Unexpected token at start of clause`.
    if let Some(index) = tokens
        .iter()
        .position(|token| matches!(token, CypherToken::BlockCommentOpen))
    {
        let (line, col) = char_offset_to_line_col(input, positions[index]);
        return Err(KgError::CypherSyntax {
            message: format_parse_error_message(input, BLOCK_COMMENT_UNSUPPORTED, line, col),
            line: Some(line),
            col: Some(col),
        });
    }
    let mut parser = CypherParser::with_keyword_lexemes(tokens, keyword_lexemes);
    match parser.parse_query() {
        Ok(q) => Ok(q),
        Err(e) => {
            let char_offset = positions
                .get(parser.pos)
                .copied()
                .unwrap_or_else(|| input.chars().count());
            let (line, col) = char_offset_to_line_col(input, char_offset);
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

/// Hook for reframing a parser error as "feature not yet implemented".
/// Always `None`: the candidates it was written for (NULLS, datetime
/// accessors, variable-length paths) all parse today, so there is nothing
/// left to detect.
fn intent_level_rewrite(_input: &str, _err: &str) -> Option<String> {
    None
}

/// Build the human-readable parse-error message body. The (line, col)
/// is included in the message text *and* carried as struct fields on
/// `KgError::CypherSyntax`; the duplication is intentional so the
/// raw message printed by `Display` is still self-contained.
fn format_parse_error_message(input: &str, err: &str, line: usize, col: usize) -> String {
    let intent = intent_level_rewrite(input, err);

    // A single-line excerpt + caret marker, rather than the whole query.
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

#[cfg(test)]
#[path = "parser_tests.rs"]
mod parser_tests;
