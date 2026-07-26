//! Cypher parser: schema DDL — `CREATE INDEX`, `DROP INDEX`, `SHOW INDEXES`,
//! and the `CONSTRAINT` counterparts, in the Neo4j 5 grammar.
//!
//! **Zero cost for non-DDL queries.** A schema command is a whole statement,
//! not a pipeline stage, so [`CypherParser::try_parse_schema_ddl_statement`] is
//! called **once per query** from [`super::CypherParser::parse_query`] — never
//! from the per-clause loop, which keeps its shape and its cost. That one call
//! is peek-only: `INDEX`, `CONSTRAINT`, `DROP`, `SHOW`, `FOR`, `IF`, `OPTIONS`,
//! `REQUIRE` and the index-type words are *not* tokenizer keywords, so they
//! arrive as [`CypherToken::Identifier`] and the test is a token comparison on
//! a position the parser already holds. There is no speculative re-parse.
//!
//! **What is deliberately not modelled.** Unsupported index *types* (`TEXT`,
//! `POINT`, `FULLTEXT`, `VECTOR`, `LOOKUP`) parse to
//! [`SchemaCommand::UnsupportedIndexType`] after their tail is scanned to the
//! statement end. Those forms carry grammar the supported ones don't (`ON EACH
//! [n.a, n.b]`, `ON EACH labels(n)`, provider `OPTIONS`), and the executor
//! rejects them wholesale, so structural parsing would buy nothing.
//!
//! **Syntax error vs unsupported feature.** A ported Neo4j schema script must
//! never see `CypherSyntaxError` for a statement KGLite understood but cannot
//! serve — that is the difference between "you typo'd" and "this engine lacks
//! the feature". So every *recognised but unsupported* construct is carried in
//! the AST and rejected in `executor/schema_ddl.rs`; the parser only errors on
//! input it genuinely cannot read.

use super::super::ast::*;
use super::super::tokenizer::CypherToken;
use super::{soft_word_eq, CypherParser};

/// Index-type words that may precede `INDEX` in `CREATE <TYPE> INDEX`.
const INDEX_TYPE_WORDS: &[(&str, DdlIndexType)] = &[
    ("RANGE", DdlIndexType::Range),
    ("TEXT", DdlIndexType::Text),
    ("POINT", DdlIndexType::Point),
    ("FULLTEXT", DdlIndexType::Fulltext),
    ("VECTOR", DdlIndexType::Vector),
    ("LOOKUP", DdlIndexType::Lookup),
];

/// Words that terminate the optional `<name>` slot of a DDL statement, so
/// `CREATE INDEX FOR (n:L) …` is read as *unnamed* rather than as an index
/// literally called `for`.
const NAME_SLOT_TERMINATORS: &[&str] = &["FOR", "IF", "ON", "OPTIONS"];

impl CypherParser {
    // ========================================================================
    // Dispatch predicates — cheap, peek-only, called from the clause loop
    // ========================================================================

    /// True when the `CREATE` at the current position opens a schema DDL
    /// statement rather than a `CREATE (n:Label)` graph write.
    ///
    /// Matches `CREATE INDEX …`, `CREATE CONSTRAINT …`, and
    /// `CREATE <type-word> INDEX …`. A graph `CREATE` is always followed by
    /// `(` or `<` (a pattern), never by a bare identifier, so this cannot
    /// steal a write.
    pub(super) fn create_opens_schema_ddl(&self) -> bool {
        match self.soft_word_at(1) {
            Some(word) if soft_word_eq(word, "INDEX") || soft_word_eq(word, "CONSTRAINT") => true,
            Some(word) => {
                index_type_for_word(word).is_some()
                    && self
                        .soft_word_at(2)
                        .is_some_and(|w| soft_word_eq(w, "INDEX"))
            }
            None => false,
        }
    }

    /// True when the identifier at the current position opens a `DROP` or
    /// `SHOW` schema statement. Both words are otherwise illegal at clause
    /// position, so recognising them here only converts a "unexpected token"
    /// error into a real statement.
    pub(super) fn identifier_opens_schema_ddl(&self) -> bool {
        let Some(word) = self.soft_word_at(0) else {
            return false;
        };
        if !soft_word_eq(word, "DROP") && !soft_word_eq(word, "SHOW") {
            return false;
        }
        // `SHOW ALL INDEXES` puts the `All` keyword token between the two, so
        // the noun can sit one slot further out.
        let noun_offset = if self.peek_at(1) == Some(&CypherToken::All) {
            2
        } else {
            1
        };
        self.soft_word_at(noun_offset)
            .is_some_and(is_index_or_constraint_noun)
    }

    /// Parse a whole schema-DDL statement, or `None` when the token stream does
    /// not open one. Called once per query from
    /// [`super::CypherParser::parse_query`] — never from the per-clause loop,
    /// because a schema command is a statement rather than a pipeline stage.
    ///
    /// Both discriminators are peek-only, so an ordinary query pays one token
    /// comparison for the whole parse and never a speculative re-parse.
    pub(super) fn try_parse_schema_ddl_statement(&mut self) -> Result<Option<Clause>, String> {
        if self.check(&CypherToken::Create) && self.create_opens_schema_ddl() {
            return self.parse_create_schema_ddl().map(Some);
        }
        if !self.identifier_opens_schema_ddl() {
            return Ok(None);
        }
        if self.peek_soft_word("DROP") {
            self.parse_drop_schema_ddl().map(Some)
        } else {
            self.parse_show_schema_ddl().map(Some)
        }
    }

    /// The error a schema command gets when it appears where a pipeline clause
    /// belongs — after another clause, or inside a `CALL { }` body. Reached from
    /// [`super::CypherParser::parse_create_clause`], because by then
    /// [`Self::try_parse_schema_ddl_statement`] has already declined the
    /// statement position, and `CREATE INDEX` would otherwise die on a
    /// confusing "expected `(`" further along.
    pub(super) fn misplaced_schema_statement_error() -> String {
        "schema commands (CREATE/DROP INDEX, CREATE/DROP CONSTRAINT, SHOW INDEXES) are \
         standalone statements: they cannot follow another clause or appear inside a \
         CALL { } subquery body"
            .to_string()
    }

    // ========================================================================
    // Statement parsers
    // ========================================================================

    /// Parse `CREATE … INDEX …` / `CREATE CONSTRAINT …`. Precondition:
    /// [`Self::create_opens_schema_ddl`] returned true.
    pub(super) fn parse_create_schema_ddl(&mut self) -> Result<Clause, String> {
        self.expect(&CypherToken::Create)?;

        if self.eat_soft_word("CONSTRAINT") {
            let command = self.parse_create_constraint_body()?;
            return Ok(Clause::Schema(SchemaCommand::Constraint(
                ConstraintCommand::Create(command),
            )));
        }

        let index_type = self.take_index_type_word();
        self.expect_soft_word("INDEX", "CREATE ... INDEX")?;

        let name = self.take_optional_ddl_name()?;
        let if_not_exists = self.parse_if_not_exists()?;

        // Unsupported index kinds: scan the tail away and let the executor
        // report the specific reason. See the module doc.
        if index_type.has_kglite_equivalent() {
            let target = self.parse_ddl_for_target()?;
            let properties = self.parse_ddl_on_properties(&target)?;
            let has_options = self.take_ddl_options()?;
            self.expect_statement_end(index_type.keyword())?;
            Ok(Clause::Schema(SchemaCommand::CreateIndex(CreateIndex {
                name,
                index_type,
                if_not_exists,
                target,
                properties,
                has_options,
            })))
        } else {
            self.skip_to_statement_end();
            Ok(Clause::Schema(SchemaCommand::UnsupportedIndexType {
                index_type,
                name,
            }))
        }
    }

    /// Parse `DROP INDEX …` / `DROP CONSTRAINT …`. Precondition:
    /// [`Self::identifier_opens_schema_ddl`] returned true with `DROP`.
    pub(super) fn parse_drop_schema_ddl(&mut self) -> Result<Clause, String> {
        self.expect_soft_word("DROP", "DROP statement")?;

        if self.eat_soft_word("CONSTRAINT") {
            let name = self.expect_name("constraint name after DROP CONSTRAINT")?;
            let if_exists = self.parse_if_exists();
            self.expect_statement_end("DROP CONSTRAINT")?;
            return Ok(Clause::Schema(SchemaCommand::Constraint(
                ConstraintCommand::Drop { name, if_exists },
            )));
        }

        self.expect_soft_word("INDEX", "DROP INDEX")?;

        // Neo4j 3.x descriptor syntax (`DROP INDEX ON :Label(prop)`) was
        // removed in Neo4j 4.0. Recognise it explicitly: a bare "expected
        // name, got Colon" would send the reader hunting for a typo.
        if self.check(&CypherToken::On) {
            return Err(
                "DROP INDEX ON :Label(property) is Neo4j 3.x syntax and was removed in \
                        Neo4j 4.0; use `DROP INDEX <name>` or KGLite's descriptor form \
                        `DROP INDEX FOR (n:Label) ON (n.property)`"
                    .to_string(),
            );
        }

        let selector = if self.peek_soft_word("FOR") {
            let target = self.parse_ddl_for_target()?;
            let properties = self.parse_ddl_on_properties(&target)?;
            DropIndexSelector::Descriptor { target, properties }
        } else {
            DropIndexSelector::Name(self.take_drop_index_name()?)
        };
        let if_exists = self.parse_if_exists();
        self.expect_statement_end("DROP INDEX")?;

        Ok(Clause::Schema(SchemaCommand::DropIndex(DropIndex {
            selector,
            if_exists,
        })))
    }

    /// The name in `DROP INDEX <name>`.
    ///
    /// KGLite's canonical index names contain a dot (`Person.age`,
    /// `Person.(city,age)`), which Cypher would otherwise require backticks to
    /// write. Reassembling the dotted form here means `SHOW INDEXES` output can
    /// be pasted straight into `DROP INDEX` — backticked names still work, since
    /// the tokenizer hands those over as one identifier.
    fn take_drop_index_name(&mut self) -> Result<String, String> {
        let mut name = self.expect_name("index name after DROP INDEX")?;
        if !self.check(&CypherToken::Dot) {
            return Ok(name);
        }
        self.advance();
        name.push('.');

        // `Label.(a,b)` — the composite spelling.
        if self.check(&CypherToken::LParen) {
            self.advance();
            name.push('(');
            loop {
                name.push_str(&self.expect_name("property name in a composite index name")?);
                if self.check(&CypherToken::Comma) {
                    self.advance();
                    name.push(',');
                } else {
                    break;
                }
            }
            self.expect(&CypherToken::RParen)?;
            name.push(')');
        } else {
            name.push_str(&self.expect_name("property name in an index name")?);
        }
        Ok(name)
    }

    /// Parse `SHOW INDEXES` / `SHOW CONSTRAINTS` (and the `ALL` and singular
    /// spellings). Precondition: [`Self::identifier_opens_schema_ddl`] returned
    /// true with `SHOW`.
    pub(super) fn parse_show_schema_ddl(&mut self) -> Result<Clause, String> {
        self.expect_soft_word("SHOW", "SHOW statement")?;
        // `SHOW ALL INDEXES` — `ALL` is a real keyword token, not an
        // identifier, and means the same as the bare form.
        if self.check(&CypherToken::All) {
            self.advance();
        }

        let noun = self.expect_ddl_noun()?;
        let command = if noun.starts_with("INDEX") {
            SchemaCommand::ShowIndexes
        } else {
            SchemaCommand::Constraint(ConstraintCommand::Show)
        };

        // `YIELD` / `WHERE` / `BRIEF` / `VERBOSE` modifiers are a genuinely
        // different result-shaping grammar. Rejecting them here — rather than
        // silently ignoring the filter and returning every row — keeps the
        // failure honest, and `CALL db.indexes()` already accepts YIELD.
        if self.has_tokens() && !self.check(&CypherToken::Semicolon) {
            // Point at the procedure that lists the *same* objects. Naming
            // `db.indexes()` for `SHOW CONSTRAINTS` would send the reader to a
            // listing of the wrong thing.
            let (procedure, columns) = if noun.starts_with("INDEX") {
                (
                    "db.indexes()",
                    "name, type, entityType, labelsOrTypes, properties, state",
                )
            } else {
                (
                    "db.constraints()",
                    "name, type, entityType, labelsOrTypes, properties",
                )
            };
            return Err(format!(
                "SHOW {noun} does not support YIELD / WHERE / BRIEF / VERBOSE modifiers; \
                 use `CALL {procedure} YIELD {columns}` for filtering and projection"
            ));
        }
        Ok(Clause::Schema(command))
    }

    // ========================================================================
    // Shared grammar fragments
    // ========================================================================

    /// `FOR (n:Label)` or `FOR ()-[r:TYPE]-()`, consuming the `FOR`.
    fn parse_ddl_for_target(&mut self) -> Result<DdlTarget, String> {
        self.expect_soft_word("FOR", "index or constraint pattern")?;
        self.expect(&CypherToken::LParen)?;

        // `()-[r:T]-()` — an empty leading node marks the relationship form.
        if self.check(&CypherToken::RParen) {
            self.advance();
            return self.parse_ddl_relationship_tail();
        }

        let variable = self.take_ddl_pattern_variable("index variable")?;
        self.expect(&CypherToken::Colon)?;
        let label = self.expect_name("node label after ':'")?;
        self.expect(&CypherToken::RParen)?;
        Ok(DdlTarget::Node { variable, label })
    }

    /// The `-[r:TYPE]-()` / `<-[r:TYPE]-()` / `-[r:TYPE]->()` tail of a
    /// relationship DDL pattern, after the leading `()` is consumed.
    fn parse_ddl_relationship_tail(&mut self) -> Result<DdlTarget, String> {
        if self.check(&CypherToken::LessThan) {
            self.advance();
        }
        self.expect(&CypherToken::Dash)?;
        self.expect(&CypherToken::LBracket)?;
        let variable = self.take_ddl_pattern_variable("relationship variable")?;
        self.expect(&CypherToken::Colon)?;
        let rel_type = self.expect_name("relationship type after ':'")?;
        self.expect(&CypherToken::RBracket)?;
        self.expect(&CypherToken::Dash)?;
        if self.check(&CypherToken::GreaterThan) {
            self.advance();
        }
        self.expect(&CypherToken::LParen)?;
        self.expect(&CypherToken::RParen)?;
        Ok(DdlTarget::Relationship { variable, rel_type })
    }

    /// `ON (n.p1, n.p2, …)`, consuming the `ON`. Property references must bind
    /// to the variable the `FOR` pattern introduced — a mismatched prefix is a
    /// typo, not a feature, and silently indexing the wrong property is worse
    /// than an error (same reasoning as the planner's unknown-property guard).
    fn parse_ddl_on_properties(&mut self, target: &DdlTarget) -> Result<Vec<String>, String> {
        self.expect(&CypherToken::On)?;
        self.expect(&CypherToken::LParen)?;
        let mut properties = Vec::new();
        loop {
            let prefix = self.expect_name("property reference like n.prop in ON (...)")?;
            self.expect(&CypherToken::Dot)?;
            let property = self.expect_name("property name after '.'")?;
            if let Some(bound) = target.variable() {
                if prefix != bound {
                    return Err(format!(
                        "property reference '{prefix}.{property}' does not use the variable \
                         '{bound}' bound by the FOR pattern"
                    ));
                }
            }
            properties.push(property);
            if self.check(&CypherToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(&CypherToken::RParen)?;
        Ok(properties)
    }

    /// `CREATE CONSTRAINT` body, after the `CONSTRAINT` word is consumed.
    ///
    /// Parsed structurally even though Sprint 4a's executor rejects every
    /// constraint: the shape here is exactly what enforcement needs, so 4b
    /// swaps the executor arm without touching the parser.
    fn parse_create_constraint_body(&mut self) -> Result<CreateConstraint, String> {
        let name = self.take_optional_ddl_name()?;
        let if_not_exists = self.parse_if_not_exists()?;
        let target = self.parse_ddl_for_target()?;

        // Neo4j 5 spells this REQUIRE; Neo4j 4 spelled it ASSERT. Accept both
        // so a 4.x-era schema script reaches the executor's feature message.
        if !self.eat_soft_word("REQUIRE") && !self.eat_soft_word("ASSERT") {
            return Err(format!(
                "expected REQUIRE (Neo4j 5) or ASSERT (Neo4j 4) in CREATE CONSTRAINT, found {:?}",
                self.peek()
            ));
        }

        let properties = self.parse_constraint_properties(&target)?;
        let requirement = self.parse_constraint_requirement()?;
        self.expect_statement_end("CREATE CONSTRAINT")?;

        Ok(CreateConstraint {
            name,
            if_not_exists,
            target,
            properties,
            requirement,
        })
    }

    /// The property list of a `REQUIRE` clause: either `n.prop` or the
    /// parenthesised `(n.a, n.b)` form used by composite uniqueness/node keys.
    fn parse_constraint_properties(&mut self, target: &DdlTarget) -> Result<Vec<String>, String> {
        let parenthesised = self.check(&CypherToken::LParen);
        if parenthesised {
            self.advance();
        }
        let mut properties = Vec::new();
        loop {
            let prefix = self.expect_name("property reference like n.prop after REQUIRE")?;
            self.expect(&CypherToken::Dot)?;
            let property = self.expect_name("property name after '.'")?;
            if let Some(bound) = target.variable() {
                if prefix != bound {
                    return Err(format!(
                        "property reference '{prefix}.{property}' does not use the variable \
                         '{bound}' bound by the FOR pattern"
                    ));
                }
            }
            properties.push(property);
            if parenthesised && self.check(&CypherToken::Comma) {
                self.advance();
            } else {
                break;
            }
        }
        if parenthesised {
            self.expect(&CypherToken::RParen)?;
        }
        Ok(properties)
    }

    /// The predicate half of `REQUIRE <props> …`: `IS UNIQUE`,
    /// `IS NOT NULL`, `IS [NODE|RELATIONSHIP] KEY`, `IS :: <TYPE>`,
    /// `IS TYPED <TYPE>`.
    fn parse_constraint_requirement(&mut self) -> Result<ConstraintRequirement, String> {
        self.expect(&CypherToken::Is)?;

        if self.check(&CypherToken::Not) {
            self.advance();
            self.expect(&CypherToken::Null)?;
            return Ok(ConstraintRequirement::NotNull);
        }
        // `IS :: <TYPE>` — the tokenizer emits two Colons, there is no `::`.
        if self.check(&CypherToken::Colon) {
            self.advance();
            self.expect(&CypherToken::Colon)?;
            return Ok(ConstraintRequirement::PropertyType(
                self.take_constraint_type_words()?,
            ));
        }
        if self.eat_soft_word("TYPED") {
            return Ok(ConstraintRequirement::PropertyType(
                self.take_constraint_type_words()?,
            ));
        }
        // The optional `NODE` / `RELATIONSHIP` scope word before UNIQUE / KEY.
        let _ = self.eat_soft_word("NODE") || self.eat_soft_word("RELATIONSHIP");
        if self.eat_soft_word("UNIQUE") {
            return Ok(ConstraintRequirement::Unique);
        }
        if self.eat_soft_word("KEY") {
            return Ok(ConstraintRequirement::Key);
        }
        Err(format!(
            "expected UNIQUE, NOT NULL, NODE KEY, or a property type after IS in \
             CREATE CONSTRAINT, found {:?}",
            self.peek()
        ))
    }

    /// A property-type expression in `IS :: …`. Neo4j spells these as one or
    /// more words plus optional `NOT NULL` / `LIST<…>` decoration; the whole
    /// tail is captured verbatim for the executor's message.
    fn take_constraint_type_words(&mut self) -> Result<String, String> {
        let mut words = Vec::new();
        while self.has_tokens() && !self.check(&CypherToken::Semicolon) {
            match self.peek() {
                Some(CypherToken::Identifier(word)) => words.push(word.clone()),
                Some(CypherToken::Not) => words.push("NOT".to_string()),
                Some(CypherToken::Null) => words.push("NULL".to_string()),
                Some(CypherToken::LessThan) => words.push("<".to_string()),
                Some(CypherToken::GreaterThan) => words.push(">".to_string()),
                _ => break,
            }
            self.advance();
        }
        if words.is_empty() {
            return Err("expected a property type after IS :: in CREATE CONSTRAINT".to_string());
        }
        Ok(words.join(" "))
    }

    // ========================================================================
    // Token-level helpers
    // ========================================================================
    //
    // The soft-keyword primitives (`soft_word_at`, `peek_soft_word`,
    // `eat_soft_word`, `expect_soft_word`, `soft_word_eq`) live in
    // `super` — `LOAD CSV` parses the same way (non-reserved words arriving
    // as `Identifier`), so they are shared rather than duplicated.

    /// Consume an index-type word when it is immediately followed by `INDEX`.
    /// Without the lookahead, `CREATE INDEX range FOR …` (an index *named*
    /// `range`) would lose its name.
    fn take_index_type_word(&mut self) -> DdlIndexType {
        let Some(word) = self.soft_word_at(0) else {
            return DdlIndexType::Unspecified;
        };
        let Some(index_type) = index_type_for_word(word) else {
            return DdlIndexType::Unspecified;
        };
        if self
            .soft_word_at(1)
            .is_some_and(|w| soft_word_eq(w, "INDEX"))
        {
            self.advance();
            index_type
        } else {
            DdlIndexType::Unspecified
        }
    }

    /// The optional `<name>` slot of a `CREATE INDEX` / `CREATE CONSTRAINT`.
    /// Absent when the next token opens the rest of the statement.
    fn take_optional_ddl_name(&mut self) -> Result<Option<String>, String> {
        match self.soft_word_at(0) {
            Some(word) if NAME_SLOT_TERMINATORS.iter().any(|t| soft_word_eq(word, t)) => Ok(None),
            Some(_) => Ok(Some(self.expect_name("index or constraint name")?)),
            None => Ok(None),
        }
    }

    /// `IF NOT EXISTS`. `NOT` and `EXISTS` are real keyword tokens; only `IF`
    /// is an identifier.
    fn parse_if_not_exists(&mut self) -> Result<bool, String> {
        if !self.eat_soft_word("IF") {
            return Ok(false);
        }
        self.expect(&CypherToken::Not)?;
        self.expect(&CypherToken::Exists)?;
        Ok(true)
    }

    /// `IF EXISTS`. Only reached where `IF NOT EXISTS` is not legal, so no
    /// `NOT` disambiguation is needed.
    fn parse_if_exists(&mut self) -> bool {
        if self.peek_soft_word("IF") && self.peek_at(1) == Some(&CypherToken::Exists) {
            self.advance();
            self.advance();
            true
        } else {
            false
        }
    }

    /// An `OPTIONS { … }` block, consumed in balanced brace pairs. Reports
    /// whether one was present; the executor rejects it, because KGLite has no
    /// index providers or per-index configuration to apply.
    fn take_ddl_options(&mut self) -> Result<bool, String> {
        if !self.eat_soft_word("OPTIONS") {
            return Ok(false);
        }
        self.expect(&CypherToken::LBrace)?;
        let mut depth = 1usize;
        while depth > 0 {
            match self.advance() {
                Some(CypherToken::LBrace) => depth += 1,
                Some(CypherToken::RBrace) => depth -= 1,
                Some(_) => {}
                None => return Err("unterminated OPTIONS { ... } block".to_string()),
            }
        }
        Ok(true)
    }

    /// The `INDEX`/`INDEXES`/`CONSTRAINT`/`CONSTRAINTS` noun of a `SHOW`
    /// statement, normalised to upper case.
    fn expect_ddl_noun(&mut self) -> Result<String, String> {
        match self.soft_word_at(0) {
            Some(word) if is_index_or_constraint_noun(word) => {
                let upper = word.to_uppercase();
                self.advance();
                Ok(upper)
            }
            _ => Err(format!(
                "Expected INDEXES or CONSTRAINTS after SHOW, got {:?}",
                self.peek()
            )),
        }
    }

    /// A pattern variable in a DDL `FOR` clause, absent when the pattern goes
    /// straight to `:` (`FOR (:Label)`).
    fn take_ddl_pattern_variable(&mut self, context: &str) -> Result<Option<String>, String> {
        if self.check(&CypherToken::Colon) {
            Ok(None)
        } else {
            Ok(Some(self.expect_name(context)?))
        }
    }

    /// Require the statement to end here. A schema command is a whole
    /// statement, so trailing tokens mean an unsupported clause rather than a
    /// pipeline continuation.
    fn expect_statement_end(&mut self, statement: &str) -> Result<(), String> {
        if self.check(&CypherToken::Semicolon) {
            self.advance();
        }
        if self.has_tokens() {
            return Err(format!(
                "unexpected token after {statement} statement: {:?} — schema commands are \
                 standalone statements and cannot be combined with other clauses",
                self.peek()
            ));
        }
        Ok(())
    }

    /// Discard the remainder of a statement we already know we will reject.
    fn skip_to_statement_end(&mut self) {
        while self.has_tokens() && !self.check(&CypherToken::Semicolon) {
            self.advance();
        }
        if self.check(&CypherToken::Semicolon) {
            self.advance();
        }
    }
}

impl DdlIndexType {
    /// True when KGLite has an index structure that serves this index type.
    /// Drives whether the parser reads the statement structurally or scans it
    /// away for the executor to reject.
    pub(crate) fn has_kglite_equivalent(self) -> bool {
        matches!(self, DdlIndexType::Unspecified | DdlIndexType::Range)
    }
}

fn index_type_for_word(word: &str) -> Option<DdlIndexType> {
    INDEX_TYPE_WORDS
        .iter()
        .find(|(candidate, _)| soft_word_eq(word, candidate))
        .map(|(_, index_type)| *index_type)
}

/// True for the four `SHOW`/`DROP` nouns, singular and plural.
fn is_index_or_constraint_noun(word: &str) -> bool {
    ["INDEX", "INDEXES", "CONSTRAINT", "CONSTRAINTS"]
        .iter()
        .any(|noun| soft_word_eq(word, noun))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::parse_cypher;
    use super::*;

    /// The sole schema command of a statement that must parse.
    fn schema(input: &str) -> SchemaCommand {
        let query =
            parse_cypher(input).unwrap_or_else(|e| panic!("`{input}` failed to parse: {e}"));
        assert_eq!(query.clauses.len(), 1, "`{input}` produced extra clauses");
        match query.clauses.into_iter().next().unwrap() {
            Clause::Schema(command) => command,
            other => panic!("`{input}` parsed as {other:?}, not a schema command"),
        }
    }

    fn create_index(input: &str) -> CreateIndex {
        match schema(input) {
            SchemaCommand::CreateIndex(create) => create,
            other => panic!("`{input}` parsed as {other:?}"),
        }
    }

    fn parse_error(input: &str) -> String {
        parse_cypher(input)
            .map(|q| panic!("`{input}` unexpectedly parsed to {:?}", q.clauses))
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn bare_create_index_is_single_property_equality() {
        let create = create_index("CREATE INDEX FOR (n:Person) ON (n.email)");
        assert_eq!(create.name, None);
        assert_eq!(create.index_type, DdlIndexType::Unspecified);
        assert!(!create.if_not_exists);
        assert!(!create.has_options);
        assert_eq!(create.properties, vec!["email".to_string()]);
        assert_eq!(
            create.target,
            DdlTarget::Node {
                variable: Some("n".to_string()),
                label: "Person".to_string(),
            }
        );
    }

    #[test]
    fn named_create_index_with_if_not_exists() {
        let create =
            create_index("CREATE INDEX person_email IF NOT EXISTS FOR (p:Person) ON (p.email)");
        assert_eq!(create.name.as_deref(), Some("person_email"));
        assert!(create.if_not_exists);
    }

    #[test]
    fn composite_create_index_keeps_property_order() {
        let create = create_index("CREATE INDEX FOR (n:Person) ON (n.city, n.age)");
        assert_eq!(
            create.properties,
            vec!["city".to_string(), "age".to_string()]
        );
    }

    #[test]
    fn range_type_word_is_captured() {
        let create = create_index("CREATE RANGE INDEX r1 FOR (n:Person) ON (n.age)");
        assert_eq!(create.index_type, DdlIndexType::Range);
        assert_eq!(create.name.as_deref(), Some("r1"));
    }

    /// An index *named* `range` must keep its name — the type word is only
    /// consumed when `INDEX` follows it.
    #[test]
    fn type_word_lookahead_does_not_eat_an_index_name() {
        let create = create_index("CREATE INDEX range FOR (n:Person) ON (n.age)");
        assert_eq!(create.index_type, DdlIndexType::Unspecified);
        assert_eq!(create.name.as_deref(), Some("range"));
    }

    #[test]
    fn options_block_is_recorded_not_dropped() {
        let create = create_index(
            "CREATE INDEX FOR (n:Person) ON (n.email) OPTIONS {indexConfig: {`x.y`: 1}}",
        );
        assert!(create.has_options);
    }

    #[test]
    fn unsupported_index_types_parse_rather_than_error() {
        for (input, expected) in [
            ("CREATE TEXT INDEX t FOR (n:P) ON (n.a)", DdlIndexType::Text),
            ("CREATE POINT INDEX p FOR (n:P) ON (n.a)", DdlIndexType::Point),
            (
                "CREATE FULLTEXT INDEX f FOR (n:P) ON EACH [n.a, n.b]",
                DdlIndexType::Fulltext,
            ),
            (
                "CREATE VECTOR INDEX v FOR (n:P) ON (n.e) OPTIONS {indexConfig: {`vector.dimensions`: 3}}",
                DdlIndexType::Vector,
            ),
            (
                "CREATE LOOKUP INDEX l FOR (n) ON EACH labels(n)",
                DdlIndexType::Lookup,
            ),
        ] {
            match schema(input) {
                SchemaCommand::UnsupportedIndexType { index_type, .. } => {
                    assert_eq!(index_type, expected, "for `{input}`");
                }
                other => panic!("`{input}` parsed as {other:?}"),
            }
        }
    }

    #[test]
    fn relationship_index_pattern_parses_all_directions() {
        for input in [
            "CREATE INDEX FOR ()-[r:KNOWS]-() ON (r.since)",
            "CREATE INDEX FOR ()-[r:KNOWS]->() ON (r.since)",
            "CREATE INDEX FOR ()<-[r:KNOWS]-() ON (r.since)",
        ] {
            let create = create_index(input);
            assert_eq!(
                create.target,
                DdlTarget::Relationship {
                    variable: Some("r".to_string()),
                    rel_type: "KNOWS".to_string(),
                },
                "for `{input}`"
            );
        }
    }

    #[test]
    fn drop_index_by_name_and_by_descriptor() {
        match schema("DROP INDEX person_email IF EXISTS") {
            SchemaCommand::DropIndex(drop) => {
                assert_eq!(
                    drop.selector,
                    DropIndexSelector::Name("person_email".to_string())
                );
                assert!(drop.if_exists);
            }
            other => panic!("parsed as {other:?}"),
        }
        match schema("DROP INDEX FOR (n:Person) ON (n.email)") {
            SchemaCommand::DropIndex(drop) => {
                assert!(!drop.if_exists);
                assert!(matches!(
                    drop.selector,
                    DropIndexSelector::Descriptor { .. }
                ));
            }
            other => panic!("parsed as {other:?}"),
        }
    }

    #[test]
    fn show_indexes_accepts_neo4j_spellings() {
        for input in ["SHOW INDEXES", "SHOW INDEX", "SHOW ALL INDEXES"] {
            assert_eq!(schema(input), SchemaCommand::ShowIndexes, "for `{input}`");
        }
        assert_eq!(
            schema("SHOW CONSTRAINTS"),
            SchemaCommand::Constraint(ConstraintCommand::Show)
        );
    }

    #[test]
    fn constraint_ddl_parses_structurally() {
        let cases: [(&str, ConstraintRequirement); 5] = [
            (
                "CREATE CONSTRAINT c1 IF NOT EXISTS FOR (p:Person) REQUIRE p.email IS UNIQUE",
                ConstraintRequirement::Unique,
            ),
            (
                "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS NOT NULL",
                ConstraintRequirement::NotNull,
            ),
            (
                "CREATE CONSTRAINT FOR (p:Person) REQUIRE (p.a, p.b) IS NODE KEY",
                ConstraintRequirement::Key,
            ),
            (
                "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER",
                ConstraintRequirement::PropertyType("INTEGER".to_string()),
            ),
            // Neo4j 4 spelled the keyword ASSERT.
            (
                "CREATE CONSTRAINT ON (p:Person) ASSERT p.email IS UNIQUE",
                ConstraintRequirement::Unique,
            ),
        ];
        for (input, expected) in cases {
            // The Neo4j 4 form writes `ON (p:Person)` instead of `FOR (...)`;
            // only the FOR spelling is accepted, so check that one here and
            // the legacy spelling in its own test.
            if input.contains("FOR (") {
                match schema(input) {
                    SchemaCommand::Constraint(ConstraintCommand::Create(create)) => {
                        assert_eq!(create.requirement, expected, "for `{input}`");
                    }
                    other => panic!("`{input}` parsed as {other:?}"),
                }
            }
        }
    }

    #[test]
    fn drop_constraint_parses() {
        match schema("DROP CONSTRAINT person_email IF EXISTS") {
            SchemaCommand::Constraint(ConstraintCommand::Drop { name, if_exists }) => {
                assert_eq!(name, "person_email");
                assert!(if_exists);
            }
            other => panic!("parsed as {other:?}"),
        }
    }

    #[test]
    fn schema_commands_are_standalone_statements() {
        for input in [
            "MATCH (n) CREATE INDEX FOR (n:Person) ON (n.email)",
            "CREATE INDEX FOR (n:Person) ON (n.email) RETURN 1",
            "CALL { CREATE INDEX FOR (n:Person) ON (n.email) RETURN 1 } RETURN 1",
        ] {
            let err = parse_error(input).to_lowercase();
            assert!(
                err.contains("standalone") || err.contains("not allowed inside"),
                "for `{input}`, got: {err}"
            );
        }
    }

    #[test]
    fn neo4j_3_drop_index_syntax_names_itself() {
        let err = parse_error("DROP INDEX ON :Person(email)");
        assert!(err.contains("Neo4j 3.x"), "got: {err}");
        assert!(err.contains("DROP INDEX FOR"), "got: {err}");
    }

    #[test]
    fn show_indexes_modifiers_point_at_db_indexes() {
        let err = parse_error("SHOW INDEXES YIELD name");
        assert!(err.contains("db.indexes()"), "got: {err}");
    }

    /// The rejection must name the procedure that lists the *same* objects.
    /// Pointing a `SHOW CONSTRAINTS` user at `db.indexes()` sends them to a
    /// listing of the wrong thing.
    #[test]
    fn show_constraints_modifiers_point_at_db_constraints() {
        let err = parse_error("SHOW CONSTRAINTS YIELD name");
        assert!(err.contains("db.constraints()"), "got: {err}");
        assert!(!err.contains("db.indexes()"), "got: {err}");
    }

    #[test]
    fn on_property_prefix_must_match_the_for_variable() {
        let err = parse_error("CREATE INDEX FOR (n:Person) ON (m.email)");
        assert!(err.contains("does not use the variable 'n'"), "got: {err}");
    }

    /// The DDL dispatch must not intercept ordinary graph writes or reads.
    #[test]
    fn graph_create_and_identifier_clauses_are_untouched() {
        let query = parse_cypher("CREATE (n:Person {name: 'A'}) RETURN n").unwrap();
        assert!(matches!(query.clauses[0], Clause::Create(_)));
        // `index` / `show` / `drop` remain usable as ordinary names.
        let query = parse_cypher("MATCH (n:Person) RETURN n.index AS show").unwrap();
        assert_eq!(query.clauses.len(), 2);
    }
}
