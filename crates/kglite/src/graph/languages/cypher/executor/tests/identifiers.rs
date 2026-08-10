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

fn run_semantics_query(graph: &mut DirGraph, query: &str) -> CypherResult {
    let parsed = parser::parse_cypher(query)
        .unwrap_or_else(|e| panic!("query failed to parse: {query}\n  error: {e}"));
    execute_mutable(
        graph,
        &parsed,
        HashMap::new(),
        crate::graph::algorithms::Interrupt::default(),
    )
    .unwrap_or_else(|e| panic!("query failed: {query}\n  error: {e}"))
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
    assert_eq!(
        tokenize_cypher("`` ` ``").unwrap_err().is_empty(),
        false,
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
