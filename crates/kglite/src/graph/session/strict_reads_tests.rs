//! `lock_schema()` rejects absent-property *reads*, not just writes.
//!
//! The lock is documented as the opt-in "catch my typos" mechanism, and it
//! already refused `MATCH (p:Person {agee: 1})` (pattern-literal property) and
//! `MATCH (p:Persn)` (unknown label). The two shapes it did *not* refuse are
//! the two an LLM or a hurried human writes most: `WHERE p.agee = 1`, which
//! returns an empty result, and `RETURN p.agee`, which returns a column of
//! nulls beside correct-looking sibling columns. Both were warnings only.
//!
//! A second family joined it: a comparison a **write-enforced** `IS :: T`
//! declaration makes vacuous (`WHERE p.age > 'forty'` on an `INTEGER` age)
//! returns an empty result for the same indistinguishable-from-data reason.
//! That one promotes only where the declaration is enforced — a
//! `define_schema()` field type is intent nothing checks at write time, so it
//! stays a warning in both schema states, and the two halves are asserted side
//! by side below.
//!
//! This module pins the promotions and — more importantly — pins that they
//! promote **nothing else**. A false positive here is a valid query that now
//! fails, so every conservatism rule the warning families were built with is
//! re-asserted through the locked path, end to end rather than at the
//! collector: `session::prepare` is where the disposition is chosen, so that
//! is where the behaviour has to be true.

use std::collections::HashMap;

use super::execute::{execute_mut, execute_read, ExecuteOptions};
use crate::datatypes::Value;
use crate::error::KgError;
use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::{NodeSchemaDefinition, SchemaDefinition, SchemaInstall};

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
    run_with(graph, query, &empty_params())
}

/// [`run`] with caller-supplied bindings — the declared-type family classifies
/// `$name` through them, so its cases need a way to bind one.
fn run_with(
    graph: &DirGraph,
    query: &str,
    params: &HashMap<String, Value>,
) -> Result<Vec<String>, String> {
    let opts = ExecuteOptions::eager(params);
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
    // ...and the declared-type promotion, pinned to the same behaviour: all
    // three are decided in `prepare`, which EXPLAIN never gets past.
    assert!(
        strict_error(
            &declared_locked(),
            "EXPLAIN MATCH (p:Person) WHERE p.age > 'forty' RETURN p"
        )
        .contains("declared INTEGER"),
        "a declared-type mismatch under a lock must reject EXPLAIN too"
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

// ── The declared-type promotion ─────────────────────────────────────────────
//
// One family, two type sources, one promotable. The pairs below are always
// written as pairs: a case that only showed the rejection would pass equally
// well in a build that promoted the unenforced half too, which is the failure
// mode that turns a lock into a nuisance.

/// [`seeded`] with DDL declarations behind `Person.age` and `Person.email` —
/// the write-enforced source, so both are promotable and a property pair
/// across them has two strong sides.
fn declared() -> DirGraph {
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let mut graph = seeded();
    for statement in [
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER",
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.email IS :: STRING",
    ] {
        execute_mut(&mut graph, statement, &opts).expect("declare the property type");
    }
    graph
}

fn declared_locked() -> DirGraph {
    let mut graph = declared();
    graph.schema_locked = true;
    graph
}

/// [`seeded`] with `define_schema()` field types instead: the same claim from
/// the source the write path does not police.
fn schema_defined() -> DirGraph {
    let mut graph = seeded();
    let mut node = NodeSchemaDefinition::default();
    for (property, declared) in [("age", "integer"), ("email", "string")] {
        node.field_types
            .insert(property.to_string(), declared.to_string());
    }
    let mut schema = SchemaDefinition::default();
    schema.node_schemas.insert("Person".to_string(), node);
    graph
        .set_schema(schema, SchemaInstall::Replace)
        .expect("the seeded rows honour the declaration");
    graph
}

fn schema_defined_locked() -> DirGraph {
    let mut graph = schema_defined();
    graph.schema_locked = true;
    graph
}

/// One side each: `age` declared by DDL, `email` only by `define_schema()`.
/// The graph the property-pair rule is decided on.
fn mixed_locked() -> DirGraph {
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let mut graph = schema_defined();
    execute_mut(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER",
        &opts,
    )
    .expect("declare the property type");
    graph.schema_locked = true;
    graph
}

const CROSS_TYPE: &str = "MATCH (p:Person) WHERE p.age > 'forty' RETURN p";

#[test]
fn locked_schema_rejects_a_declared_type_mismatch() {
    let message = strict_error(&declared_locked(), CROSS_TYPE);
    // The family's own wording, verbatim — one sentence describing the
    // mistake, whichever disposition reports it.
    assert!(
        message.contains("WHERE compares Person.age (declared INTEGER)"),
        "{message}"
    );
    assert!(
        message.contains("with a STRING literal 'forty'"),
        "{message}"
    );
    assert!(
        message.contains("a cross-type ordering comparison is null in openCypher"),
        "{message}"
    );
    // ...and the lock's own suffix, identical to the absent-property one, so a
    // reader who hits either learns the same way out.
    assert!(
        message.contains(
            "(the schema is locked — call unlock_schema() to make this a warning instead)"
        ),
        "{message}"
    );
}

/// The other source, under the same lock, on the same query text: a warning,
/// and the query runs.
#[test]
fn a_schema_defined_type_mismatch_stays_a_warning_under_the_lock() {
    let graph = schema_defined_locked();
    let warnings = accepted(&graph, CROSS_TYPE);
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("Person.age (schema-defined integer)")),
        "{warnings:?}"
    );
    // Not merely "did not raise": the statement executed and answered.
    let rows = execute_read(
        &graph,
        "MATCH (p:Person) WHERE p.age <> 'forty' RETURN p.age",
        &ExecuteOptions::eager(&empty_params()),
    )
    .expect("an unpromotable finding never stops the query")
    .result
    .rows;
    assert_eq!(rows.len(), 2, "{rows:?}");
}

#[test]
fn an_open_schema_only_warns_about_either_source() {
    for graph in [declared(), schema_defined()] {
        let warnings = accepted(&graph, CROSS_TYPE);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains("filters out every row"),
            "{warnings:?}"
        );
    }
}

#[test]
fn unlocking_restores_the_declared_type_mismatch_warning() {
    let mut graph = declared_locked();
    strict_error(&graph, CROSS_TYPE);
    graph.schema_locked = false;
    let warnings = accepted(&graph, CROSS_TYPE);
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("Person.age (declared INTEGER)")),
        "{warnings:?}"
    );
}

/// The cache-exclusion sequence, exactly as the absent-property family pins
/// it: a plan primed while the schema was open must not carry its pre-lock
/// verdict past the lock. Flipping the flag directly does not bump the graph
/// version, so the entry would still be live if one had been stored.
#[test]
fn a_declared_type_plan_primed_before_the_lock_is_still_rejected_after_it() {
    let mut graph = declared();
    for _ in 0..2 {
        accepted(&graph, CROSS_TYPE);
    }
    let version_before = graph.version();
    graph.schema_locked = true;
    assert_eq!(
        graph.version(),
        version_before,
        "this test is about a lock that did NOT invalidate the cache"
    );
    assert!(strict_error(&graph, CROSS_TYPE).contains("declared INTEGER"));
}

/// A pair is only as strong as its weaker side.
#[test]
fn a_property_pair_promotes_only_when_both_sides_are_declared() {
    const PAIR: &str = "MATCH (p:Person) WHERE p.age > p.email RETURN p";
    let message = strict_error(&declared_locked(), PAIR);
    assert!(
        message.contains("Person.age (declared INTEGER)"),
        "{message}"
    );
    assert!(
        message.contains("Person.email (declared STRING)"),
        "{message}"
    );

    let warnings = accepted(&mixed_locked(), PAIR);
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("Person.email (schema-defined string)")),
        "one unenforced side must leave the finding a warning: {warnings:?}"
    );
}

/// A bound `$param` promotes against a declared property: the property side is
/// write-enforced and the parameter's type is a fact of this call, so the
/// predicate is exactly as unsatisfiable as its literal spelling — and rather
/// more deserving of an error, since the query text does not show the mistake.
/// The verdict is per call, which is why the well-typed binding is asserted in
/// the same test.
#[test]
fn a_bound_parameter_promotes_against_a_declared_type() {
    let graph = declared_locked();
    let query = "MATCH (p:Person) WHERE p.age > $cutoff RETURN p";

    let mut bad = HashMap::new();
    bad.insert("cutoff".to_string(), Value::String("forty".to_string()));
    let message = match run_with(&graph, query, &bad) {
        Err(message) => message,
        Ok(warnings) => panic!("expected a schema error, got warnings {warnings:?}"),
    };
    assert!(
        message.contains("a STRING parameter $cutoff ('forty')"),
        "{message}"
    );
    assert!(message.contains("unlock_schema()"), "{message}");

    let mut good = HashMap::new();
    good.insert("cutoff".to_string(), Value::Int64(35));
    let warnings = run_with(&graph, query, &good).expect("a well-typed binding is not a mistake");
    assert!(warnings.is_empty(), "{warnings:?}");
    // An *unbound* name is no knowledge, so the lock adds nothing to it: the
    // statement still fails with the executor's own "Missing parameter", not a
    // schema error. Silence at the collector is pinned in
    // `type_mismatch::tests::an_unbound_parameter_says_nothing`.
}

/// Every site that can produce a finding also decides its own disposition —
/// `IN`, the string predicates and `=~`, not just the comparison path.
#[test]
fn every_finding_site_promotes_a_declared_source_and_only_that() {
    for query in [
        "MATCH (p:Person) WHERE p.age IN ['a', 'b'] RETURN p",
        "MATCH (p:Person) WHERE p.age STARTS WITH 'x' RETURN p",
        "MATCH (p:Person) WHERE p.age CONTAINS 'x' RETURN p",
        "MATCH (p:Person) WHERE p.age =~ 'x.*' RETURN p",
    ] {
        assert!(
            strict_error(&declared_locked(), query).contains("unlock_schema()"),
            "{query}"
        );
        let warnings = accepted(&schema_defined_locked(), query);
        assert_eq!(warnings.len(), 1, "{query} -> {warnings:?}");
    }
}

/// The conservatisms, through the locked path: a comparison the runtime can
/// answer is never rejected, whatever the declaration says.
#[test]
fn well_typed_and_undeclared_comparisons_are_never_rejected() {
    let graph = declared_locked();
    for query in [
        // The declared family and the literal's family agree.
        "MATCH (p:Person) WHERE p.age > 30 RETURN p",
        // INTEGER and FLOAT are one comparison family, all nine pairings live.
        "MATCH (p:Person) WHERE p.age > 30.5 RETURN p",
        // A string predicate on the STRING declaration.
        "MATCH (p:Person) WHERE p.email STARTS WITH 'a' RETURN p",
        // No declaration behind the property at all — observed metadata is not
        // a source, so there is nothing to promote.
        "MATCH (a:Paper) WHERE a.year > 'x' RETURN a",
        // A built-in field reads the title, not a declared property.
        "MATCH (p:Person) WHERE p.name > 5 RETURN p",
        // No label to resolve a declaration from.
        "MATCH (n) WHERE n.age > 'forty' RETURN n",
    ] {
        assert!(accepted(&graph, query).is_empty(), "{query}");
    }
}

/// `WITH`-rebound and multi-label variables are absent from the label map, so
/// this family inherits the absent-property family's silence about them —
/// re-asserted here because a lock turning one of them into a hard error is
/// the expensive kind of false positive.
#[test]
fn unresolvable_variables_are_never_rejected_by_the_type_family() {
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let graph = declared_locked();
    accepted(
        &graph,
        "MATCH (p:Person) WITH p AS q WHERE q.age > 'forty' RETURN q",
    );

    let mut graph = declared();
    execute_mut(&mut graph, "MATCH (p:Person) SET p:Admin", &opts).expect("secondary label");
    graph.schema_locked = true;
    accepted(
        &graph,
        "MATCH (p:Person:Admin) WHERE p.age > 'forty' RETURN p",
    );
}
