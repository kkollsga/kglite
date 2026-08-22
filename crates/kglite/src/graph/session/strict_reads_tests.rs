//! `lock_schema()` rejects absent-property *reads*, not just writes.
//!
//! The lock is documented as the opt-in "catch my typos" mechanism, and it
//! already refused `MATCH (p:Person {agee: 1})` (pattern-literal property) and
//! `MATCH (p:Persn)` (unknown label). The two shapes it did *not* refuse are
//! the two an LLM or a hurried human writes most: `WHERE p.agee = 1`, which
//! returns an empty result, and `RETURN p.agee`, which returns a column of
//! nulls beside correct-looking sibling columns. Both were warnings only.
//!
//! This module pins the promotion and — more importantly — pins that it
//! promotes **nothing else**. A false positive here is a valid query that now
//! fails, so every conservatism rule the warning families were built with is
//! re-asserted through the locked path, end to end rather than at the
//! collector: `session::prepare` is where the disposition is chosen, so that
//! is where the behaviour has to be true.

use std::collections::HashMap;

use super::execute::{execute_mut, execute_read, ExecuteOptions};
use crate::datatypes::Value;
use crate::error::KgError;
use crate::graph::dir_graph::DirGraph;

fn empty_params() -> HashMap<String, Value> {
    HashMap::new()
}

/// `Person {age, email}` (email sparse — one of the two people has none),
/// `Paper {year}`, `AUTHORED` running Person→Paper, and `Tag`, whose only
/// values are built-ins so its property metadata is empty.
fn seeded() -> DirGraph {
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let mut graph = DirGraph::new();
    for statement in [
        "CREATE (:Person {id: 1, age: 30, email: 'a@b.c'})-[:AUTHORED]->(:Paper {id: 2, year: 2020})",
        "CREATE (:Person {id: 3, age: 40})",
        "CREATE (:Tag {id: 4})",
    ] {
        execute_mut(&mut graph, statement, &opts).expect("seed write");
    }
    graph
}

fn locked() -> DirGraph {
    let mut graph = seeded();
    graph.schema_locked = true;
    graph
}

/// `Ok(the query's warnings)` or `Err(the schema error's message)`. Any other
/// failure panics where it happens: nothing in this module has a second way to
/// fail, and narrowing the error to a `String` keeps `KgError` — 128 bytes of
/// `ValidationError` at its widest — out of a helper's return type.
fn run(graph: &DirGraph, query: &str) -> Result<Vec<String>, String> {
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    match execute_read(graph, query, &opts) {
        Ok(outcome) => Ok(outcome
            .result
            .diagnostics
            .expect("every execution carries diagnostics")
            .warnings),
        Err(KgError::Schema { message, .. }) => Err(message),
        Err(other) => panic!("{query}: unexpected error {other:?}"),
    }
}

/// The error message for a query that must be rejected as a locked-schema
/// absent-property read.
fn strict_error(graph: &DirGraph, query: &str) -> String {
    match run(graph, query) {
        Err(message) => message,
        Ok(warnings) => panic!("{query}: expected a schema error, got warnings {warnings:?}"),
    }
}

/// Assert `query` runs under the lock. Returns its warnings so a caller can
/// also assert the finding survived *as* a warning.
fn accepted(graph: &DirGraph, query: &str) -> Vec<String> {
    match run(graph, query) {
        Ok(warnings) => warnings,
        Err(message) => panic!("{query}: locked schema wrongly rejected this — {message}"),
    }
}

// ── The promotion ───────────────────────────────────────────────────────────

#[test]
fn locked_schema_rejects_an_absent_property_in_where() {
    let message = strict_error(&locked(), "MATCH (p:Person) WHERE p.agee = 1 RETURN p");
    assert!(
        message.contains("Unknown property 'agee' on Person, referenced in WHERE"),
        "{message}"
    );
    assert!(message.contains("Did you mean 'age'?"), "{message}");
    // The valid set and the escape hatch, so the message is actionable without
    // a trip to describe() or the docs.
    assert!(
        message.contains("Valid properties: age, email"),
        "{message}"
    );
    assert!(message.contains("unlock_schema()"), "{message}");
}

#[test]
fn locked_schema_rejects_an_absent_property_in_return() {
    let message = strict_error(&locked(), "MATCH (p:Person) RETURN p.name, p.agee");
    assert!(
        message.contains("Unknown property 'agee' on Person, referenced in RETURN"),
        "{message}"
    );
}

#[test]
fn locked_schema_rejects_absent_properties_in_with_and_order_by() {
    let graph = locked();
    assert!(
        strict_error(&graph, "MATCH (p:Person) WITH p.agee AS a RETURN a")
            .contains("referenced in WITH"),
    );
    assert!(
        strict_error(&graph, "MATCH (p:Person) RETURN p ORDER BY p.agee")
            .contains("referenced in ORDER BY"),
    );
}

/// A mutation reaches the same `prepare`, so a typo in the `WHERE` that
/// *selects the rows to write* is rejected before anything is written.
#[test]
fn locked_schema_rejects_an_absent_property_read_inside_a_mutation() {
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let mut graph = locked();
    let err = match execute_mut(
        &mut graph,
        "MATCH (p:Person) WHERE p.agee = 1 SET p.age = 99",
        &opts,
    ) {
        Err(err) => err,
        Ok(_) => panic!("a locked schema must reject the typo'd selector"),
    };
    assert!(
        matches!(&err, KgError::Schema { message, .. } if message.contains("'agee'")),
        "{err:?}"
    );
    // Nothing was written: the rejection happens at prepare, before execution.
    let rows = execute_read(&graph, "MATCH (p:Person) WHERE p.age = 99 RETURN p", &opts)
        .expect("read")
        .result
        .rows;
    assert!(rows.is_empty(), "{rows:?}");
}

/// EXPLAIN of a strict-failing query errors rather than rendering a plan —
/// matching what a locked schema already did for an unknown *label*, which is
/// also rejected in `prepare` and therefore never reaches the plan renderer.
#[test]
fn explain_of_a_strict_failing_query_errors_like_an_unknown_label_does() {
    let graph = locked();
    assert!(strict_error(&graph, "EXPLAIN MATCH (p:Person) RETURN p.agee").contains("'agee'"),);
    // The precedent, asserted in the same test so a change to either is a
    // visible divergence rather than a silent one.
    assert!(
        strict_error(&graph, "EXPLAIN MATCH (n:Persn) RETURN n").contains("Persn"),
        "unknown label under a lock must reject EXPLAIN too"
    );
}

// ── The default is untouched ────────────────────────────────────────────────

#[test]
fn an_open_schema_still_only_warns() {
    let graph = seeded();
    for query in [
        "MATCH (p:Person) WHERE p.agee = 1 RETURN p",
        "MATCH (p:Person) RETURN p.agee",
    ] {
        let warnings = accepted(&graph, query);
        assert!(
            warnings.iter().any(|w| w.contains("Did you mean 'age'?")),
            "{query} -> {warnings:?}"
        );
    }
}

#[test]
fn unlocking_restores_the_warning() {
    let mut graph = locked();
    strict_error(&graph, "MATCH (p:Person) RETURN p.agee");
    graph.schema_locked = false;
    let warnings = accepted(&graph, "MATCH (p:Person) RETURN p.agee");
    assert!(
        warnings.iter().any(|w| w.contains("Did you mean 'age'?")),
        "{warnings:?}"
    );
}

/// The plan cache returns before the schema pass runs, so a plan primed while
/// the schema was open must not let the same query text slip past the lock.
/// Locking through the `api`/Python surface bumps the graph version and
/// invalidates the entry anyway; this flips the flag directly, which does not,
/// so it pins the property that does not depend on that.
#[test]
fn a_plan_primed_before_the_lock_is_still_rejected_after_it() {
    let mut graph = seeded();
    let query = "MATCH (p:Person) RETURN p.agee";
    // Twice, so the second call would be a cache hit if the statement were
    // cacheable at all.
    for _ in 0..2 {
        accepted(&graph, query);
    }
    let version_before = graph.version();
    graph.schema_locked = true;
    assert_eq!(
        graph.version(),
        version_before,
        "this test is about a lock that did NOT invalidate the cache"
    );
    assert!(strict_error(&graph, query).contains("'agee'"));
}

// ── The conservatism rules, re-asserted through the locked path ─────────────

/// A property only *some* nodes carry is in the type's metadata, so it is not
/// absent — the sparse-column false positive that would break real pipelines.
#[test]
fn a_sparse_property_is_not_a_typo() {
    let graph = locked();
    assert!(accepted(&graph, "MATCH (p:Person) RETURN p.email, p.age").is_empty());
}

/// A type whose metadata is empty is under-declared, not wrong — the same
/// skip `validate_property` makes on the write path.
#[test]
fn a_type_with_no_property_metadata_is_never_rejected() {
    let graph = locked();
    assert!(accepted(&graph, "MATCH (t:Tag) RETURN t.anything").is_empty());
}

/// Built-ins and untyped vars: neither has a metadata entry to be absent from.
#[test]
fn builtins_and_untyped_vars_are_never_rejected() {
    let graph = locked();
    accepted(
        &graph,
        "MATCH (p:Person) RETURN p.id, p.title, p.name, p.type",
    );
    accepted(&graph, "MATCH (n) WHERE n.whatever = 1 RETURN n");
}

/// A multi-label pattern binds no single label to reason from, so the var is
/// dropped from the label map and every check stays silent about it.
#[test]
fn a_multi_label_pattern_is_never_rejected() {
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let mut graph = seeded();
    execute_mut(&mut graph, "MATCH (p:Person) SET p:Admin", &opts).expect("secondary label");
    graph.schema_locked = true;
    accepted(&graph, "MATCH (p:Person:Admin) RETURN p.agee");
}

/// `WITH n AS m` rebinds a var the label map never tracked, so `m.agee` is
/// unknowable rather than wrong.
#[test]
fn a_with_rebound_var_is_never_rejected() {
    let graph = locked();
    accepted(&graph, "MATCH (n:Person) WITH n AS m RETURN m.agee");
    accepted(
        &graph,
        "MATCH (n:Person) WITH n AS m WHERE m.agee = 1 RETURN m",
    );
}

/// A property the statement itself writes is not absent by the time it is read
/// back. Checked through EXPLAIN because the *write* of a new property is
/// separately refused by the lock at execution time (`enforce_schema_lock`) —
/// EXPLAIN returns straight after `prepare`, which is the disposition this
/// module is about.
#[test]
fn a_property_the_same_statement_writes_is_never_rejected_at_prepare() {
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let mut graph = locked();
    for query in [
        "EXPLAIN MATCH (p:Person) SET p.badprop = 1 RETURN p.badprop",
        "EXPLAIN MATCH (p:Person) SET p += {badprop: 1} RETURN p.badprop",
    ] {
        if let Err(err) = execute_mut(&mut graph, query, &opts) {
            panic!("{query}: rejected at prepare — {err:?}");
        }
    }
}

/// `OPTIONAL MATCH` is treated exactly as `MATCH` for the schema question —
/// optionality is about rows, not about whether a property name exists.
#[test]
fn optional_match_is_treated_as_match() {
    let graph = locked();
    accepted(&graph, "OPTIONAL MATCH (p:Person) RETURN p.age");
    assert!(strict_error(
        &graph,
        "OPTIONAL MATCH (p:Person) WHERE p.agee = 1 RETURN p"
    )
    .contains("'agee'"),);
}

/// The reversed-arrow family is a heuristic over observed endpoint pairs, not
/// a name check, so a lock leaves it a warning.
#[test]
fn a_reversed_arrow_stays_a_warning_under_the_lock() {
    let graph = locked();
    let warnings = accepted(&graph, "MATCH (a:Paper)-[:AUTHORED]->(p:Person) RETURN p");
    assert!(
        warnings.iter().any(|w| w.contains("Reverse the arrow?")),
        "{warnings:?}"
    );
}

/// An unknown relationship type also stays a warning: a lock covers node types
/// and their properties, and an edge type is not yet part of what it asserts.
#[test]
fn an_unknown_relationship_type_stays_a_warning_under_the_lock() {
    let graph = locked();
    let warnings = accepted(&graph, "MATCH (p:Person)-[:AUTHRED]->(a:Paper) RETURN p");
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("unknown relationship type 'AUTHRED'")),
        "{warnings:?}"
    );
}

/// An unknown *label* under a lock is rejected by `validate_label`, which runs
/// before the warning collector — so it is reported once, as the label
/// mistake, and never doubled up with a property complaint about a type that
/// does not exist.
#[test]
fn an_unknown_label_is_reported_once_as_a_label_error() {
    let message = strict_error(&locked(), "MATCH (p:Persn) WHERE p.agee = 1 RETURN p");
    assert!(message.contains("Unknown node type 'Persn'"), "{message}");
    assert!(!message.contains("agee"), "{message}");
}
