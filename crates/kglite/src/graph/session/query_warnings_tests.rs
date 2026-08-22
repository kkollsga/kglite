//! `QueryDiagnostics.warnings` is populated by the engine, on every path.
//!
//! The field existed since the diagnostics struct landed and was never set by
//! `execute_read`/`execute_mut`: the schema warnings went to **stderr only**,
//! and the wheel re-derived them separately. Every non-Python surface (MCP,
//! Bolt, C ABI, any Rust consumer) therefore saw an empty `warnings` list on a
//! query that the engine had *already* diagnosed as a typo.
//!
//! The trap this module exists for is the **plan cache**. `prepare()` returns
//! early on a cache hit, before the schema pass runs — computing the warnings
//! there would force the parse the cache exists to skip, and *not* computing
//! them silently drops the warning on the second and every later run of the
//! same query. The warnings ride on the cache entry instead: they are a pure
//! function of `(query, graph schema)` and the key already pins the graph
//! state, so a hit hands back exactly what a miss would have computed.

use std::collections::HashMap;

use super::execute::{execute_mut, execute_read, ExecuteOptions};
use crate::datatypes::Value;
use crate::graph::dir_graph::DirGraph;
use crate::graph::languages::cypher::plan_cache;
use crate::graph::languages::cypher::plan_cache::instrumentation;
use crate::graph::schema::{NodeSchemaDefinition, SchemaDefinition, SchemaInstall};

fn empty_params() -> HashMap<String, Value> {
    HashMap::new()
}

/// A graph with one node type (`Vessel`) and one edge type (`OPERATED_BY`),
/// so both the unknown-label and unknown-relationship checks have candidates
/// to suggest from.
fn seeded() -> DirGraph {
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let mut graph = DirGraph::new();
    execute_mut(
        &mut graph,
        "CREATE (:Vessel {id: 1})-[:OPERATED_BY]->(:Operator {id: 2})",
        &opts,
    )
    .expect("seed write");
    graph
}

fn warnings_of(graph: &DirGraph, query: &str) -> Vec<String> {
    warnings_with(graph, query, &empty_params())
}

fn warnings_with(graph: &DirGraph, query: &str, params: &HashMap<String, Value>) -> Vec<String> {
    let opts = ExecuteOptions::eager(params);
    let outcome = execute_read(graph, query, &opts).expect("read");
    outcome
        .result
        .diagnostics
        .expect("every execution carries diagnostics")
        .warnings
}

#[test]
fn unknown_label_reaches_diagnostics() {
    let graph = seeded();
    let warnings = warnings_of(&graph, "MATCH (n:vessel) RETURN count(n) AS c");
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("unknown node label 'vessel'") && w.contains("Did you mean")),
        "case-typo label must reach diagnostics: {warnings:?}"
    );
}

#[test]
fn unknown_relationship_reaches_diagnostics() {
    let graph = seeded();
    let warnings = warnings_of(&graph, "MATCH (a:Vessel)-[:OPERATED_BYY]->(b) RETURN a");
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("unknown relationship type 'OPERATED_BYY'")),
        "{warnings:?}"
    );
}

#[test]
fn a_clean_query_carries_diagnostics_with_no_warnings() {
    let graph = seeded();
    let warnings = warnings_of(&graph, "MATCH (n:Vessel) RETURN count(n) AS c");
    assert!(warnings.is_empty(), "{warnings:?}");
}

/// **The cache trap.** The second run of the same query is served from the
/// plan cache, which returns before the schema pass — the warning must
/// survive that path, and the hit must be real (asserted from the
/// instrumentation counters, so this cannot pass by silently re-parsing).
#[test]
fn a_plan_cache_hit_still_carries_its_warnings() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let graph = seeded();
    let query = "MATCH (n:vessel) RETURN count(n) AS c";

    instrumentation::reset();
    let first = warnings_of(&graph, query);
    let second = warnings_of(&graph, query);
    let stats = instrumentation::totals().read;

    assert_eq!(
        stats.hits, 1,
        "the second run must be a cache hit, else this case proves nothing: {stats:?}"
    );
    assert!(!first.is_empty(), "first run: {first:?}");
    assert_eq!(
        first, second,
        "a cache hit must carry the same warnings as the miss that filled it"
    );
}

/// A mutation's *read* patterns are diagnosed too — `MATCH (n:typo) SET …`
/// silently updates nothing, the same foot-gun in write clothing. (Write
/// patterns stay undiagnosed on purpose: `CREATE (n:NewType)` is how a type
/// comes into existence.)
#[test]
fn a_mutations_read_pattern_reaches_diagnostics() {
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let mut graph = seeded();
    let outcome =
        execute_mut(&mut graph, "MATCH (n:vessel) SET n.flag = true", &opts).expect("mut");
    let warnings = outcome
        .result
        .diagnostics
        .expect("a mutation carries diagnostics too")
        .warnings;
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("unknown node label 'vessel'")),
        "{warnings:?}"
    );

    let created = execute_mut(&mut graph, "CREATE (:BrandNewType {id: 9})", &opts).expect("mut");
    assert!(
        created
            .result
            .diagnostics
            .expect("diagnostics")
            .warnings
            .is_empty(),
        "a CREATE of an unseen type is not a typo"
    );
}

/// A procedure's scoping values are validated at *execution* time, deep inside
/// the executor, so their warnings cannot come from `prepare`. They ride out
/// on the same field.
#[test]
fn procedure_scope_warnings_reach_diagnostics() {
    let graph = seeded();
    let warnings = warnings_of(
        &graph,
        "CALL pagerank({relationship: 'OPERATED_BYY'}) YIELD node RETURN count(*) AS c",
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("unknown relationship type 'OPERATED_BYY'")),
        "{warnings:?}"
    );
}

/// EXPLAIN renders a plan without executing; the typo that would have made the
/// plan return nothing is exactly what the reader is looking for.
#[test]
fn explain_carries_warnings() {
    let graph = seeded();
    let warnings = warnings_of(&graph, "EXPLAIN MATCH (n:vessel) RETURN n");
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("unknown node label 'vessel'")),
        "{warnings:?}"
    );
}

/// A `CALL {}` body runs on its own executor. Its warnings must be absorbed
/// by the outer one, or they die with the sub-executor and the caller reads a
/// clean result for a query that was scoped at nothing.
#[test]
fn a_subquery_bodys_procedure_warning_is_absorbed() {
    let graph = seeded();
    let warnings = warnings_of(
        &graph,
        "CALL { CALL pagerank({relationship: 'OPERATED_BYY'}) YIELD node \
         RETURN count(*) AS c } RETURN c",
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("unknown relationship type 'OPERATED_BYY'")),
        "{warnings:?}"
    );
}

/// A **correlated** `CALL {}` re-runs its body once per outer row, so the same
/// warning is re-discovered every time. It is one fact about the query and is
/// reported once — a per-row list would bury the result it is meant to explain.
#[test]
fn a_correlated_subquerys_warning_is_absorbed_once() {
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let mut graph = seeded();
    execute_mut(&mut graph, "CREATE (:Vessel {id: 3})", &opts).expect("second vessel");
    let warnings = warnings_of(
        &graph,
        "MATCH (v:Vessel) CALL { WITH v CALL pagerank({relationship: 'OPERATED_BYY'}) \
         YIELD node RETURN count(*) AS c } RETURN v.id, c",
    );
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0].contains("unknown relationship type 'OPERATED_BYY'"),
        "{warnings:?}"
    );
}

/// The declared-type family (`WHERE p.age > 'forty'` on an `IS :: INTEGER`
/// property) reaches diagnostics on **every** run — but not through the cache.
/// `lock_schema()` promotes this half of the family to a `SchemaError`, so the
/// statement is excluded from the cache exactly as an absent-property one is,
/// and a hit's inability to re-decide anything never comes up.
#[test]
fn a_declared_type_mismatch_reaches_diagnostics_and_is_never_cached() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let mut graph = DirGraph::new();
    execute_mut(&mut graph, "CREATE (:Person {id: 1, age: 30})", &opts).expect("seed");
    execute_mut(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER",
        &opts,
    )
    .expect("declare the property type");

    let query = "MATCH (p:Person) WHERE p.age > 'forty' RETURN p";
    instrumentation::reset();
    let first = warnings_of(&graph, query);
    let second = warnings_of(&graph, query);
    let stats = instrumentation::totals().read;

    assert_eq!(
        (stats.insertions, stats.hits),
        (0, 0),
        "a promotable finding must keep the statement out of the cache: {stats:?}"
    );
    assert_eq!(first.len(), 1, "{first:?}");
    assert!(
        first[0].contains("Person.age (declared INTEGER)")
            && first[0].contains("STRING literal 'forty'")
            && first[0].contains("filters out every row"),
        "{first:?}"
    );
    assert_eq!(
        first, second,
        "the second, uncached run must recompute the same warning"
    );

    // An undeclared property of the same type is untouched by the family.
    assert!(
        warnings_of(&graph, "MATCH (p:Person) WHERE p.id > 'forty' RETURN p").is_empty(),
        "a built-in field carries no declaration"
    );
}

/// The other half of the family, and the reason the exclusion is written as a
/// *subset* rather than a family: a `define_schema()` field type promotes
/// nowhere, so nothing about it can go stale behind a lock and its plan is
/// cached like any other read's — warning included, served off the entry.
///
/// This is the control for the case above. Without it, "the declared one is not
/// cached" would also pass in a build where the family disabled caching
/// wholesale, or where nothing was cacheable at all.
#[test]
fn a_schema_defined_type_mismatch_still_rides_the_cache() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let params = empty_params();
    let opts = ExecuteOptions::eager(&params);
    let mut graph = DirGraph::new();
    execute_mut(&mut graph, "CREATE (:Person {id: 1, age: 30})", &opts).expect("seed");
    let mut node = NodeSchemaDefinition::default();
    node.field_types
        .insert("age".to_string(), "integer".to_string());
    let mut schema = SchemaDefinition::default();
    schema.node_schemas.insert("Person".to_string(), node);
    graph
        .set_schema(schema, SchemaInstall::Replace)
        .expect("the stored ages honour the declaration");

    let query = "MATCH (p:Person) WHERE p.age > 'forty' RETURN p";
    instrumentation::reset();
    let first = warnings_of(&graph, query);
    let second = warnings_of(&graph, query);
    let stats = instrumentation::totals().read;

    assert_eq!(
        (stats.insertions, stats.hits),
        (1, 1),
        "an unpromotable finding must not cost the statement its cache entry: {stats:?}"
    );
    assert_eq!(first.len(), 1, "{first:?}");
    assert!(
        first[0].contains("Person.age (schema-defined integer)"),
        "{first:?}"
    );
    assert_eq!(first, second, "the hit must carry the same warning");
}

/// A `$param` is classified through the **caller's** bindings, and the answer
/// cannot go stale in the plan cache: a statement with bound parameters is
/// never cached at all (`prepare`'s `cacheable` gate), and the empty-params
/// invocation of the same text — the one that *is* cached — binds nothing and
/// therefore says nothing about the parameter.
#[test]
fn a_parameter_typed_mismatch_is_diagnosed_and_never_cached() {
    let _guard = plan_cache::TEST_LOCK
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let empty = empty_params();
    let opts = ExecuteOptions::eager(&empty);
    let mut graph = DirGraph::new();
    execute_mut(&mut graph, "CREATE (:Person {id: 1, age: 30})", &opts).expect("seed");
    execute_mut(
        &mut graph,
        "CREATE CONSTRAINT FOR (p:Person) REQUIRE p.age IS :: INTEGER",
        &opts,
    )
    .expect("declare the property type");

    let query = "MATCH (p:Person) WHERE p.age > $cutoff RETURN p";
    let mut bound = HashMap::new();
    bound.insert("cutoff".to_string(), Value::String("forty".to_string()));

    instrumentation::reset();
    let first = warnings_with(&graph, query, &bound);
    assert_eq!(first.len(), 1, "{first:?}");
    assert!(
        first[0].contains("Person.age (declared INTEGER)")
            && first[0].contains("STRING parameter $cutoff ('forty')"),
        "{first:?}"
    );
    // Same text, well-typed binding: the finding is a fact about the *value*,
    // so it must not survive from the run before it.
    let mut typed = HashMap::new();
    typed.insert("cutoff".to_string(), Value::Int64(40));
    assert!(warnings_with(&graph, query, &typed).is_empty());
    assert!(warnings_with(&graph, query, &bound).len() == 1);

    let stats = instrumentation::totals().read;
    assert_eq!(
        stats.hits, 0,
        "a parameterized statement is never cached, so nothing can be served stale: {stats:?}"
    );
}
