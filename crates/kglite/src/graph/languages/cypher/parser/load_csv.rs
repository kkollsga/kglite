//! Cypher parser: `LOAD CSV`, in the spelling other Cypher databases use.
//!
//! ```text
//! LOAD CSV [WITH HEADERS] FROM <expression> AS <variable>
//!     [FIELDTERMINATOR <single-character string>]
//! ```
//!
//! **Zero cost for other queries.** `LOAD`, `CSV`, `HEADERS`, `FROM` and
//! `FIELDTERMINATOR` are deliberately *not* tokenizer keywords — reserving
//! them would break every graph holding a property, label, or alias of the
//! same name. They arrive as [`CypherToken::Identifier`] and are matched with
//! the shared soft-keyword helpers in [`super`], so the clause loop pays one
//! extra identifier comparison and only when a clause actually starts with a
//! bare identifier.
//!
//! `WITH` *is* a keyword, so `WITH HEADERS` is [`CypherToken::With`] followed
//! by the identifier `HEADERS`. That is unambiguous: a real `WITH` projection
//! clause can never appear between `LOAD CSV` and `FROM`.
//!
//! **Leading position only.** `LOAD CSV` originates rows from outside the
//! graph, and the executor drives the remaining clauses over bounded batches
//! of those rows (`executor/load_csv.rs`). Both properties only hold when it
//! is the first clause, so a misplaced `LOAD CSV` is rejected with
//! [`CypherParser::misplaced_load_csv_error`] rather than parsed and quietly
//! mis-executed.
//!
//! **Syntax error vs unsupported feature.** Same rule as schema DDL: a ported
//! script must never see a syntax error for something KGLite read but cannot
//! serve. `http://` and `https://` sources parse cleanly here and are
//! rejected at execute time with the network-free-design explanation — see
//! `executor/load_csv.rs`.

use super::super::ast::*;
use super::super::tokenizer::CypherToken;
use super::CypherParser;

impl CypherParser {
    /// True when the identifier at the current position opens a `LOAD CSV`
    /// clause. Peek-only: two token comparisons, no speculative re-parse.
    ///
    /// Requires the `CSV` word as well as `LOAD`, so a query using `load` as
    /// an ordinary identifier is untouched.
    pub(super) fn identifier_opens_load_csv(&self) -> bool {
        self.peek_soft_word("LOAD")
            && self
                .soft_word_at(1)
                .is_some_and(|w| w.eq_ignore_ascii_case("CSV"))
    }

    /// The error a `LOAD CSV` gets when it appears anywhere but first.
    ///
    /// Reported instead of parsing it, because the streaming driver can only
    /// batch the pipeline it leads; accepting the clause in a later position
    /// would either silently buffer the whole file or silently drop the
    /// upstream rows.
    pub(super) fn misplaced_load_csv_error() -> String {
        "LOAD CSV must be the first clause of the query: it is a row source, and KGLite streams \
         the rest of the pipeline over batches of its rows. Move it to the front, or load the \
         file separately and pass the rows in as a parameter."
            .to_string()
    }

    /// Parse `LOAD CSV [WITH HEADERS] FROM <expr> AS <var> [FIELDTERMINATOR <sep>]`.
    /// Precondition: [`Self::identifier_opens_load_csv`] returned true.
    pub(super) fn parse_load_csv_clause(&mut self) -> Result<Clause, String> {
        self.expect_soft_word("LOAD", "LOAD CSV")?;
        self.expect_soft_word("CSV", "LOAD CSV")?;

        // `WITH HEADERS` — the `WITH` here is the keyword token.
        let with_headers = if self.check(&CypherToken::With) {
            self.advance();
            self.expect_soft_word("HEADERS", "LOAD CSV WITH HEADERS")?;
            true
        } else {
            false
        };

        self.expect_soft_word("FROM", "LOAD CSV")?;
        let source = self.parse_expression()?;

        self.expect(&CypherToken::As)?;
        let variable = self.try_consume_alias_name()?;

        let field_terminator = if self.eat_soft_word("FIELDTERMINATOR") {
            Some(self.parse_field_terminator()?)
        } else {
            None
        };

        Ok(Clause::LoadCsv(LoadCsvClause {
            with_headers,
            source,
            variable,
            field_terminator,
        }))
    }

    /// Parse the `FIELDTERMINATOR` operand: a string literal holding exactly
    /// one single-byte character.
    ///
    /// Rejecting multi-byte and multi-character separators here rather than at
    /// execute time keeps the failure next to the typo. The `csv` reader takes
    /// a `u8` delimiter, so `FIELDTERMINATOR '→'` has no representation — say
    /// so plainly instead of truncating it.
    fn parse_field_terminator(&mut self) -> Result<u8, String> {
        let literal = match self.advance().cloned() {
            Some(CypherToken::StringLit(s)) => s,
            Some(token) => {
                return Err(format!(
                    "FIELDTERMINATOR expects a quoted single-character separator, got {token:?}"
                ))
            }
            None => {
                return Err(
                    "FIELDTERMINATOR expects a quoted single-character separator, but \
                            reached end of query"
                        .to_string(),
                )
            }
        };

        let bytes = literal.as_bytes();
        match bytes.len() {
            1 => Ok(bytes[0]),
            0 => Err("FIELDTERMINATOR cannot be the empty string".to_string()),
            _ => Err(format!(
                "FIELDTERMINATOR must be a single-byte character, got {literal:?} \
                 ({} bytes). Multi-character and non-ASCII separators are not supported.",
                bytes.len()
            )),
        }
    }
}
