//! **Absolute goldens for quoted-identifier escaping — the injection class.**
//!
//! A backtick-quoted identifier had no escape: the tokenizer read to the first
//! closing backtick and stopped, so doubling did not work and an identifier
//! carrying a backtick was simply unrepresentable. That made any caller who
//! string-built a label, relationship type, property key, alias or pattern
//! variable from untrusted input injectable — the quote could be closed and
//! arbitrary clauses appended.
//!
//! Both exploits below were **executed against the shipped 0.15.9 extension**
//! before the fix, from the DSL investigation's probes:
//!
//! ```text
//! label = "Person`) DETACH DELETE n //"
//! emitted: MATCH (n:`Person`) DETACH DELETE n //`) RETURN count(n) AS c
//! result:  0 rows; every Person node deleted
//!
//! var = "n` :Secret) RETURN n.title AS leaked //"
//! emitted: MATCH (`n` :Secret) RETURN n.title AS leaked //`:Person) RETURN count(n) AS c
//! result:  [{leaked: 'classified'}] — a node the query had no business reading
//! ```
//!
//! With doubling in place, the caller escapes and the payload becomes one
//! (weird) identifier that matches nothing: the clause boundary never appears.
//! These are parser/tokenizer semantics with one right answer, so the gate is
//! absolute goldens, not the differential corpus.

use super::*;
use crate::graph::languages::cypher::tokenizer::{tokenize_cypher, CypherToken};

/// The escaping an emitter must apply: wrap in backticks, double any inside.
/// Mirrors `parser::match_pattern::backtick_quote` and Python's
/// `kglite._cypher_identifier` — this test is what keeps the three in step.
fn quote(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

fn graph_with_secrets() -> DirGraph {
    let mut graph = DirGraph::new();
    run_semantics_query(
        &mut graph,
        "CREATE (:Person {id: 1, title: 'ada'}), (:Person {id: 2, title: 'bob'}), \
         (:Secret {id: 3, title: 'classified'})",
    );
    graph
}

/// Run `query`, routing it the way production does — `is_mutation_query`
/// picks the engine. Running a read through `execute_mutable` unconditionally
/// is not a faithful harness: a MATCH-less `RETURN` answers zero rows there,
/// which would have silently voided the value-position controls below.
fn run_semantics_query(graph: &mut DirGraph, query: &str) -> CypherResult {
    let parsed = parser::parse_cypher(query)
        .unwrap_or_else(|e| panic!("query failed to parse: {query}\n  error: {e}"));
    let outcome = if crate::graph::languages::cypher::executor::write::is_mutation_query(&parsed) {
        execute_mutable(
            graph,
            &parsed,
            HashMap::new(),
            crate::graph::algorithms::Interrupt::default(),
        )
    } else {
        let no_params = HashMap::new();
        CypherExecutor::with_params(graph, &no_params, None).execute(&parsed)
    };
    outcome.unwrap_or_else(|e| panic!("query failed: {query}\n  error: {e}"))
}

fn count(graph: &mut DirGraph, label: &str) -> i64 {
    let result = run_semantics_query(graph, &format!("MATCH (n:{label}) RETURN count(n) AS c"));
    match &result.rows[0][0] {
        Value::Int64(n) => *n,
        other => panic!("expected a count, got {other:?}"),
    }
}

// ========================================================================
// Tokenizer: doubling is the escape
// ========================================================================

#[test]
fn doubled_backtick_is_one_literal_backtick() {
    // Was: `Expected RParen, found Identifier("ird")` — the quote closed at the
    // first inner backtick and `ird` fell out as grammar.
    assert_eq!(
        tokenize_cypher("`We``ird`").unwrap(),
        vec![CypherToken::Identifier("We`ird".to_string())]
    );
    // Leading, trailing, and consecutive escaped backticks.
    assert!(
        !tokenize_cypher("`` ` ``").unwrap_err().is_empty(),
        "an unterminated quoted identifier must still be an error"
    );
    assert_eq!(
        tokenize_cypher("`a````b`").unwrap(),
        vec![CypherToken::Identifier("a``b".to_string())]
    );
    assert_eq!(
        tokenize_cypher("```a`").unwrap(),
        vec![CypherToken::Identifier("`a".to_string())]
    );
    // Plain quoted identifiers are untouched.
    assert_eq!(
        tokenize_cypher("`My Node`").unwrap(),
        vec![CypherToken::Identifier("My Node".to_string())]
    );
}

#[test]
fn an_unterminated_quoted_identifier_is_still_rejected() {
    // Non-vacuity for the escape: doubling must not swallow the terminator.
    // `\u{60}abc` never closes; `\u{60}a\u{60}\u{60}b` closes nothing either.
    assert!(tokenize_cypher("`abc")
        .unwrap_err()
        .contains("Unterminated"));
    assert!(tokenize_cypher("`a``b")
        .unwrap_err()
        .contains("Unterminated"));
}

#[test]
fn quote_then_tokenize_round_trips_every_hostile_identifier() {
    for hostile in [
        "Person`) DETACH DELETE n //",
        "n` :Secret) RETURN n.title AS leaked //",
        "`",
        "``",
        "a`b`c",
        "plain",
        "with space",
        "with-hyphen.and.dots",
    ] {
        let tokens = tokenize_cypher(&quote(hostile)).unwrap_or_else(|e| {
            panic!("quoted {hostile:?} failed to tokenize: {e}");
        });
        assert_eq!(
            tokens,
            vec![CypherToken::Identifier(hostile.to_string())],
            "quote-then-tokenize must round-trip {hostile:?} as ONE identifier"
        );
    }
}

// ========================================================================
// The two reproduced exploits, now inert
// ========================================================================

#[test]
fn label_position_injection_is_inert_when_escaped() {
    let mut graph = graph_with_secrets();
    assert_eq!(count(&mut graph, "Person"), 2);

    let label = "Person`) DETACH DELETE n //";
    let query = format!("MATCH (n:{}) RETURN count(n) AS c", quote(label));
    let result = run_semantics_query(&mut graph, &query);

    // The payload is one label that matches nothing…
    assert_eq!(result.rows[0][0], Value::Int64(0));
    // …and, the whole point, the DETACH DELETE never became a clause.
    assert_eq!(
        count(&mut graph, "Person"),
        2,
        "the injected DETACH DELETE must not have run"
    );
    assert_eq!(count(&mut graph, "Secret"), 1);
}

#[test]
fn variable_position_injection_cannot_exfiltrate() {
    let mut graph = graph_with_secrets();

    let var = "n` :Secret) RETURN n.title AS leaked //";
    let query = format!(
        "MATCH ({}:Person) RETURN count({}) AS c",
        quote(var),
        quote(var)
    );
    let result = run_semantics_query(&mut graph, &query);

    // One column named `c`, not the injected `leaked`: the appended RETURN
    // never parsed as a clause. Pre-fix this answered [{leaked: 'classified'}].
    assert_eq!(result.columns, vec!["c"]);
    assert_eq!(result.rows[0][0], Value::Int64(2));
}

#[test]
fn unescaped_injection_still_breaks_out() {
    // **Non-vacuity, and the reason the two tests above mean anything** (R1).
    // The escape is a *caller* obligation; the grammar only makes it possible.
    // A caller that interpolates raw — the shape the exploit used — still
    // produces a query whose payload is grammar, and this test proves the
    // harness can see that. If this ever goes green, the tests above have
    // stopped measuring the escaping and started measuring nothing.
    let mut graph = graph_with_secrets();
    let label = "Person`) DETACH DELETE n //";
    let naive = format!("MATCH (n:`{label}`) RETURN count(n) AS c");
    run_semantics_query(&mut graph, &naive);
    assert_eq!(
        count(&mut graph, "Person"),
        0,
        "the raw-interpolation control must still be exploitable — otherwise \
         the escaped cases above are not testing the escape"
    );
}

// ========================================================================
// Emitter round-trip through the secondary pattern lexer
// ========================================================================

#[test]
fn a_backtick_bearing_label_survives_the_pattern_round_trip() {
    // `EXISTS { }` and `count { }` patterns are re-serialized from tokens and
    // re-lexed by `core::pattern_matching`, so the emitter's quoting and that
    // lexer's escape rule have to agree. A label carrying a backtick is the
    // case that catches a disagreement.
    let mut graph = DirGraph::new();
    let weird = "Od`d";
    run_semantics_query(
        &mut graph,
        &format!("CREATE (:{} {{id: 1, title: 'x'}})", quote(weird)),
    );
    let result = run_semantics_query(
        &mut graph,
        &format!("MATCH (n:{}) RETURN n.title AS t", quote(weird)),
    );
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::String("x".to_string()));

    // The property-key position too, including through EXISTS.
    run_semantics_query(
        &mut graph,
        &format!(
            "MATCH (n:{}) SET n.{} = 7",
            quote(weird),
            quote("we`ird key")
        ),
    );
    let result = run_semantics_query(
        &mut graph,
        &format!(
            "MATCH (n:{}) WHERE n.{} = 7 RETURN count(n) AS c",
            quote(weird),
            quote("we`ird key")
        ),
    );
    assert_eq!(result.rows[0][0], Value::Int64(1));
}

// ========================================================================
// The value-literal words — TRUE / FALSE / NULL — in NAME positions
// ========================================================================
//
// **The contract, and where it comes from.** openCypher 9 spells the name of a
// label, relationship type or property key as
// `SchemaName = SymbolicName | ReservedWord`, and `ReservedWord` lists TRUE,
// FALSE and NULL — so those words are legal *schema* names written bare. A
// variable is `SymbolicName` alone, which excludes them; a variable spelled
// `true` needs backticks. Neo4j 25 agrees on the schema half
// (`labelType : COLON symbolicNameString`, whose unescaped alternatives include
// TRUE / FALSE / NULL) and is *more* permissive on variables — but a bare
// `true` in an expression is the literal there too, so a variable named that
// way could never be read back, which is the same mint-but-never-query trap
// this section closes. KGLite stops at the openCypher line.
//
// The trap as reported: ``CREATE (:`TRUE` {x:1})`` succeeded and
// ``MATCH (n:`TRUE`)`` failed. The MATCH path re-serializes its token stream
// for the secondary pattern parser, and the backtick escape was destroyed in
// transit — the bare word was re-read as a boolean. Position is what decides a
// name from a value, so these are absolute goldens on both parsers, with the
// value positions carried alongside as the controls.

/// Parse-only probe — the error cases below must not reach the executor.
fn parses(query: &str) -> bool {
    parser::parse_cypher(query).is_ok()
}

#[test]
fn a_minted_reserved_word_label_can_be_queried_back() {
    // The exact reported asymmetry: creatable, unmatchable.
    let mut graph = DirGraph::new();
    run_semantics_query(&mut graph, "CREATE (:`TRUE` {id: 1, title: 'minted'})");
    let result = run_semantics_query(&mut graph, "MATCH (n:`TRUE`) RETURN n.title AS t");
    assert_eq!(
        result.rows.len(),
        1,
        "a label that can be minted must match"
    );
    assert_eq!(result.rows[0][0], Value::String("minted".to_string()));
}

#[test]
fn reserved_literal_words_are_labels_in_both_parsers() {
    let mut graph = DirGraph::new();
    for (id, label) in ["TRUE", "FALSE", "NULL"].iter().enumerate() {
        // CREATE parses labels through `expect_name`; MATCH through the
        // token re-serializer and the secondary pattern parser.
        run_semantics_query(
            &mut graph,
            &format!("CREATE (:{label} {{id: {}, title: 't'}})", id + 1),
        );
    }
    for label in ["TRUE", "FALSE", "NULL"] {
        assert_eq!(
            count(&mut graph, label),
            1,
            "bare {label} in label position"
        );
        assert_eq!(
            count(&mut graph, &format!("`{label}`")),
            1,
            "backticked {label} in label position"
        );
    }
    // Multi-label and SET/REMOVE label positions take them too.
    run_semantics_query(&mut graph, "CREATE (:Thing:NULL {id: 9})");
    assert_eq!(count(&mut graph, "NULL"), 2);
}

#[test]
fn a_reserved_literal_name_keeps_its_verbatim_case() {
    // Same rule as the other soft keywords (`names.soft_keyword_verbatim_case`):
    // the stored name is the source lexeme, so `TRUE` and `true` are two labels.
    let mut graph = DirGraph::new();
    run_semantics_query(&mut graph, "CREATE (:TRUE {id: 1}), (:true {id: 2})");
    assert_eq!(count(&mut graph, "TRUE"), 1);
    assert_eq!(count(&mut graph, "true"), 1);
}

#[test]
fn reserved_literal_words_are_relationship_types_in_both_parsers() {
    let mut graph = DirGraph::new();
    run_semantics_query(
        &mut graph,
        "CREATE (a:Node {id: 1}), (b:Node {id: 2}), (c:Node {id: 3})",
    );
    run_semantics_query(
        &mut graph,
        "MATCH (a:Node {id: 1}), (b:Node {id: 2}) CREATE (a)-[:TRUE]->(b)",
    );
    run_semantics_query(
        &mut graph,
        "MATCH (a:Node {id: 1}), (c:Node {id: 3}) CREATE (a)-[:`NULL`]->(c)",
    );

    for rel in ["TRUE", "`TRUE`"] {
        let result = run_semantics_query(
            &mut graph,
            &format!("MATCH (a:Node)-[:{rel}]->(b:Node) RETURN b.id AS id"),
        );
        assert_eq!(result.rows.len(), 1, "rel-type position {rel}");
        assert_eq!(result.rows[0][0], Value::Int64(2));
    }
    // Alternation: the type after `|` is a name position too.
    let result = run_semantics_query(
        &mut graph,
        "MATCH (a:Node)-[:TRUE|NULL]->(b:Node) RETURN count(b) AS c",
    );
    assert_eq!(result.rows[0][0], Value::Int64(2));
}

#[test]
fn reserved_literal_words_are_property_keys_in_both_parsers() {
    let mut graph = DirGraph::new();
    run_semantics_query(
        &mut graph,
        "CREATE (:Thing {id: 1, true: 7, false: 8, null: 9})",
    );
    // Inline-map KEY position in a MATCH pattern, plus dotted reads.
    let result = run_semantics_query(
        &mut graph,
        "MATCH (n:Thing {true: 7}) RETURN n.false AS f, n.null AS nu",
    );
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Int64(8));
    assert_eq!(result.rows[0][1], Value::Int64(9));
    // Backticked spells the same key.
    let result = run_semantics_query(
        &mut graph,
        "MATCH (n:Thing {`true`: 7}) RETURN n.`true` AS t",
    );
    assert_eq!(result.rows[0][0], Value::Int64(7));
    // SET and WHERE reach the key through the expression parser.
    run_semantics_query(&mut graph, "MATCH (n:Thing) SET n.null = 11");
    let result = run_semantics_query(
        &mut graph,
        "MATCH (n:Thing) WHERE n.null = 11 RETURN count(n) AS c",
    );
    assert_eq!(result.rows[0][0], Value::Int64(1));
}

#[test]
fn reserved_literal_names_survive_the_exists_subquery_re_serializer() {
    // `EXISTS { }` re-serializes through a second extractor; it has to make
    // the same name/value call as the top-level one.
    let mut graph = DirGraph::new();
    run_semantics_query(&mut graph, "CREATE (a:Node {id: 1}), (b:`NULL` {id: 2})");
    run_semantics_query(
        &mut graph,
        "MATCH (a:Node {id: 1}), (b:`NULL` {id: 2}) CREATE (a)-[:TRUE]->(b)",
    );
    let result = run_semantics_query(
        &mut graph,
        "MATCH (a:Node) WHERE EXISTS { (a)-[:TRUE]->(:NULL) } RETURN a.id AS id",
    );
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Int64(1));
}

#[test]
fn value_positions_still_read_the_literals() {
    // **The controls.** Everything above is position-sensitive acceptance, so
    // these are the cells that catch a global keyword-table fix: a value
    // position must keep reading the literal.
    let mut graph = DirGraph::new();
    run_semantics_query(
        &mut graph,
        "CREATE (:Flag {id: 1, on: true}), (:Flag {id: 2, on: false})",
    );
    // Inline-map VALUE in a MATCH pattern.
    let result = run_semantics_query(&mut graph, "MATCH (n:Flag {on: true}) RETURN n.id AS id");
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Int64(1));
    // WHERE and RETURN.
    let result = run_semantics_query(
        &mut graph,
        "MATCH (n:Flag) WHERE n.on = true RETURN n.id AS id",
    );
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Int64(1));
    let result = run_semantics_query(
        &mut graph,
        "RETURN true AS t, false AS f, null IS NULL AS n",
    );
    assert_eq!(result.rows[0][0], Value::Boolean(true));
    assert_eq!(result.rows[0][1], Value::Boolean(false));
    assert_eq!(result.rows[0][2], Value::Boolean(true));
    // A map literal holding both: key `true`, value `true`.
    let result = run_semantics_query(&mut graph, "RETURN {true: 1, x: true} AS m");
    match &result.rows[0][0] {
        Value::Map(m) => {
            assert_eq!(m.get("true"), Some(&Value::Int64(1)));
            assert_eq!(m.get("x"), Some(&Value::Boolean(true)));
        }
        other => panic!("expected a map, got {other:?}"),
    }
}

#[test]
fn a_bare_reserved_literal_is_not_a_variable_in_either_parser() {
    // openCypher's `Variable = SymbolicName` excludes the reserved words, and
    // a bare `true` in an expression is the literal — so a variable spelled
    // that way is unreadable by construction. Both parsers refuse it, which is
    // what keeps CREATE and MATCH symmetric; backticks are the escape.
    assert!(!parses("CREATE (true:Thing)"), "bare variable in CREATE");
    assert!(
        !parses("MATCH (true:Thing) RETURN 1"),
        "bare variable in MATCH"
    );

    // Backticked, it is an ordinary variable end to end — the CYPHER.md
    // example, executed.
    let mut graph = DirGraph::new();
    run_semantics_query(&mut graph, "CREATE (`true`:Thing {id: 4})");
    let result = run_semantics_query(&mut graph, "MATCH (`true`:Thing) RETURN `true`.id AS id");
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Int64(4));
}

#[test]
fn the_documented_reserved_literal_examples_run() {
    // The CYPHER.md "Reserved keywords as names" snippet, verbatim — a doc
    // example that does not execute is a claim, not a contract.
    let mut graph = DirGraph::new();
    run_semantics_query(&mut graph, "CREATE (:TRUE {null: 1})-[:FALSE]->(:Thing)");
    let result = run_semantics_query(
        &mut graph,
        "MATCH (n:TRUE {null: 1})-[:FALSE]->() RETURN n.null AS nu",
    );
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Int64(1));
}

#[test]
fn reserved_literal_names_reach_every_clause_that_names_a_label() {
    // MERGE, SET and REMOVE name labels through their own parse paths; a fix
    // applied only to MATCH/CREATE would leave them behind.
    let mut graph = DirGraph::new();
    let result = run_semantics_query(&mut graph, "MERGE (n:TRUE {id: 5}) RETURN n.id AS x");
    assert_eq!(result.rows[0][0], Value::Int64(5));
    // The second MERGE must find the first node, not mint a second — the
    // backticked and bare spellings have to name the same label.
    run_semantics_query(&mut graph, "MERGE (n:`TRUE` {id: 5})");
    assert_eq!(count(&mut graph, "TRUE"), 1);

    run_semantics_query(&mut graph, "CREATE (:Node {id: 1})");
    let result = run_semantics_query(
        &mut graph,
        "MATCH (n:Node) SET n:NULL RETURN labels(n) AS l",
    );
    assert_eq!(
        result.rows[0][0],
        Value::List(vec![
            Value::String("Node".to_string()),
            Value::String("NULL".to_string())
        ])
    );
    let result = run_semantics_query(
        &mut graph,
        "MATCH (n:Node) REMOVE n:NULL RETURN labels(n) AS l",
    );
    assert_eq!(
        result.rows[0][0],
        Value::List(vec![Value::String("Node".to_string())])
    );
}

#[test]
fn a_reserved_literal_name_and_value_coexist_in_one_subquery_pattern() {
    // The re-serializer decides name-vs-value per token, so the case that
    // catches a depth-blind fix is both in one pattern: `:TRUE` is a
    // relationship type, `{on: false}` is a boolean property.
    let mut graph = DirGraph::new();
    run_semantics_query(
        &mut graph,
        "CREATE (a:Node {id: 1, on: true})-[:TRUE]->(b:Node {id: 2, on: false})",
    );
    let result = run_semantics_query(
        &mut graph,
        "MATCH (a:Node) WHERE EXISTS { (a)-[:TRUE]->({on: false}) } RETURN a.id AS id",
    );
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0][0], Value::Int64(1));
    let result = run_semantics_query(
        &mut graph,
        "MATCH (a:Node {on: true}) RETURN COUNT { (a)-[:TRUE]->() } AS c",
    );
    assert_eq!(result.rows[0][0], Value::Int64(1));
}
