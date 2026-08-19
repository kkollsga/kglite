//! Write-time enforcement of relationship constraints at the three inline
//! choke points — `CREATE` an edge, `SET r.p`, `REMOVE r.p` — plus the
//! no-phantom contract every one of them owes the change log.
//!
//! The bulk loader's gate has its own per-conflict-mode matrix in
//! `mutation/rel_constraint_gate_tests.rs`; this file is the Cypher half.

use crate::datatypes::Value;
use crate::graph::algorithms::Interrupt;
use crate::graph::cdc;
use crate::graph::constraints::{ConstraintKind, EntityKind};
use crate::graph::dir_graph::DirGraph;
use crate::graph::property_types::DeclaredType;
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use std::collections::HashMap;

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("{query}: {e}"));
}

fn run_err(graph: &mut DirGraph, query: &str) -> String {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    match execute_mut(graph, query, &opts) {
        Ok(_) => panic!("`{query}` should have been refused"),
        Err(error) => error.to_string(),
    }
}

/// Two `Person`s joined by one `KNOWS` carrying `since = 2020`.
fn knows_graph() -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (a:Person {person_id: 1})-[:KNOWS {since: 2020}]->(b:Person {person_id: 2})",
    );
    graph
}

fn typed_graph() -> DirGraph {
    let mut graph = knows_graph();
    graph
        .create_rel_property_type_constraint(
            "KNOWS",
            "since",
            DeclaredType::Integer,
            &Interrupt::default(),
        )
        .expect("declaration over clean data");
    graph
}

fn required_graph() -> DirGraph {
    let mut graph = knows_graph();
    graph
        .create_rel_not_null_constraint("KNOWS", "since", &Interrupt::default())
        .expect("declaration over clean data");
    graph
}

fn edge_count(graph: &DirGraph) -> usize {
    use crate::graph::storage::GraphRead;
    graph.graph.edge_weights().count()
}

// ── site 1: CREATE ───────────────────────────────────────────────────

/// The A3b rule, on the relationship side: a refused CREATE must leave the
/// connection-type metadata exactly as it found it. Registering the type
/// first would teach `describe()` a relationship type — and a property shape
/// — that no successful write ever produced.
#[test]
fn a_refused_create_leaves_the_connection_type_metadata_untouched() {
    let mut graph = typed_graph();
    let before = graph.connection_type_metadata.get("KNOWS").cloned();
    let error = run_err(
        &mut graph,
        "MATCH (a:Person {person_id: 1}), (b:Person {person_id: 2}) \
         CREATE (a)-[:KNOWS {since: 'yesterday'}]->(b)",
    );
    assert!(error.contains("STRING"), "{error}");
    assert!(error.contains("relationship of type 'KNOWS'"), "{error}");
    assert_eq!(edge_count(&graph), 1, "the refused edge must not exist");

    let after = graph.connection_type_metadata.get("KNOWS").cloned();
    let property_types = |info: &Option<crate::graph::schema::ConnectionTypeInfo>| {
        info.as_ref().map(|info| {
            let mut names: Vec<String> = info.property_types.keys().cloned().collect();
            names.sort();
            names
        })
    };
    assert_eq!(
        property_types(&after),
        property_types(&before),
        "a refused CREATE must not register a property shape"
    );
}

/// A wholly new relationship type is not registered by a refused create
/// either — the case where the metadata entry would have been *created*.
#[test]
fn a_refused_create_does_not_register_a_new_connection_type() {
    let mut graph = knows_graph();
    graph
        .create_rel_not_null_constraint("RATES", "score", &Interrupt::default())
        .expect("an empty type is vacuously clean");
    let error = run_err(
        &mut graph,
        "MATCH (a:Person {person_id: 1}), (b:Person {person_id: 2}) CREATE (a)-[:RATES]->(b)",
    );
    assert!(error.contains("'score'"), "{error}");
    assert!(
        !graph.connection_type_metadata.contains_key("RATES"),
        "the refused type must stay unknown to the schema"
    );
    assert!(
        !graph.has_connection_type("RATES"),
        "and unknown to the lightweight cache too"
    );
}

/// MERGE's create branch routes through `execute_create`, so it is gated by
/// the same declaration without a second gate.
#[test]
fn merge_creates_through_the_same_gate() {
    let mut graph = typed_graph();
    run(&mut graph, "CREATE (:Person {person_id: 3})");
    // A pair with no edge between it, so MERGE must take its create branch.
    let error = run_err(
        &mut graph,
        "MATCH (a:Person {person_id: 1}), (c:Person {person_id: 3}) \
         MERGE (a)-[:KNOWS {since: 'yesterday'}]->(c)",
    );
    assert!(error.contains("STRING"), "{error}");
    assert_eq!(edge_count(&graph), 1, "the refused edge must not exist");
}

// ── site 2: SET ──────────────────────────────────────────────────────

#[test]
fn set_is_refused_by_a_declared_type() {
    let mut graph = typed_graph();
    let error = run_err(
        &mut graph,
        "MATCH ()-[r:KNOWS]->() SET r.since = 'yesterday'",
    );
    assert!(error.contains("STRING"), "{error}");
    assert!(error.contains("KNOWS.since"), "{error}");
}

/// Null is not a type mismatch but it *is* an absence, so the two kinds
/// disagree about `SET r.p = null` — exactly as they do on the node side.
#[test]
fn set_to_null_is_refused_by_presence_and_allowed_by_a_type() {
    let mut graph = required_graph();
    let error = run_err(&mut graph, "MATCH ()-[r:KNOWS]->() SET r.since = null");
    assert!(error.contains("must have the property 'since'"), "{error}");

    let mut typed = typed_graph();
    run(&mut typed, "MATCH ()-[r:KNOWS]->() SET r.since = null");
}

/// `SET r = {…}` and `SET r += {…}` desugar into the property and remove
/// items above, so gating those two branches gates every SET spelling. A gate
/// on the map branch alone would miss `SET r.p`, and one here misses nothing.
#[test]
fn map_assignment_is_gated_through_its_desugaring() {
    for query in [
        "MATCH ()-[r:KNOWS]->() SET r = {since: 'yesterday'}",
        "MATCH ()-[r:KNOWS]->() SET r += {since: 'yesterday'}",
    ] {
        let mut graph = typed_graph();
        let error = run_err(&mut graph, query);
        assert!(error.contains("STRING"), "for `{query}`: {error}");
    }
    // `SET r = {...}` drops the properties the map omits, so a required one
    // has to be refused through the REMOVE half of the same desugaring.
    let mut graph = required_graph();
    let error = run_err(&mut graph, "MATCH ()-[r:KNOWS]->() SET r = {other: 1}");
    assert!(error.contains("'since'"), "{error}");
}

// ── site 3: REMOVE ───────────────────────────────────────────────────

#[test]
fn remove_is_refused_by_presence_and_allowed_by_a_type() {
    let mut graph = required_graph();
    let error = run_err(&mut graph, "MATCH ()-[r:KNOWS]->() REMOVE r.since");
    assert!(error.contains("must have the property 'since'"), "{error}");
    assert!(error.contains("relationship"), "{error}");

    // A declared *type* says nothing about presence: removing is legal.
    let mut typed = typed_graph();
    run(&mut typed, "MATCH ()-[r:KNOWS]->() REMOVE r.since");
}

// ── typed errors and the fast-out ────────────────────────────────────

/// The structured violation rides the side channel so a binding raises its
/// typed error rather than re-parsing prose.
#[test]
fn the_violation_reaches_the_caller_typed() {
    let mut graph = required_graph();
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let error = match execute_mut(&mut graph, "MATCH ()-[r:KNOWS]->() REMOVE r.since", &opts) {
        Ok(_) => panic!("a required property cannot be removed"),
        Err(error) => error,
    };
    match error {
        crate::error::KgError::ConstraintViolation {
            kind,
            node_type,
            properties,
            descriptor,
            ..
        } => {
            assert_eq!(kind, "NOT NULL");
            assert_eq!(node_type, "KNOWS");
            assert_eq!(properties, vec!["since".to_string()]);
            assert_eq!(descriptor, "KNOWS.since");
        }
        other => panic!("expected the typed constraint error, got {other:?}"),
    }
}

/// The same journey at the engine's own `Result<_, String>` boundary: the
/// structured violation is parked under the exact message, which is what the
/// session's conversion above matches on.
#[test]
fn the_violation_is_parked_under_its_message() {
    use super::super::super::parser::parse_cypher;
    let mut graph = required_graph();
    let parsed = parse_cypher("MATCH ()-[r:KNOWS]->() REMOVE r.since").unwrap();
    let error = super::execute_mutable(&mut graph, &parsed, HashMap::new(), Interrupt::default())
        .expect_err("a required property cannot be removed");
    let violation = graph
        .take_constraint_violation_for(&error)
        .expect("the violation must be parked under its own message");
    assert_eq!(violation.kind, ConstraintKind::NotNull);
    assert_eq!(violation.entity, EntityKind::Relationship);
    assert_eq!(violation.node_type, "KNOWS");
}

/// The fast-out's observable shape: an unconstrained graph — and a graph that
/// constrains some *other* relationship type — resolves no connection type at
/// all, which is what keeps an unconstrained SET from reading the edge and
/// the interner.
#[test]
fn an_unconstrained_write_resolves_no_relationship_type() {
    use super::super::edge_property_write::constrained_edge_type;
    let mut graph = knows_graph();
    let edge = petgraph::graph::EdgeIndex::new(0);
    assert!(
        constrained_edge_type(&graph, edge).is_none(),
        "a graph that declares nothing must not resolve the edge's type"
    );

    graph
        .create_rel_not_null_constraint("RATES", "score", &Interrupt::default())
        .expect("declaration on another type");
    assert!(
        constrained_edge_type(&graph, edge).is_none(),
        "a constraint on another type must not make this edge pay for it"
    );

    graph
        .create_rel_not_null_constraint("KNOWS", "since", &Interrupt::default())
        .expect("declaration on this type");
    assert_eq!(
        constrained_edge_type(&graph, edge).as_deref(),
        Some("KNOWS"),
        "the type is resolved only once something constrains it"
    );
}

// ── no phantoms in the change log ────────────────────────────────────

/// Every gate sits ahead of the call that captures, so a refused write leaves
/// the capture buffer empty. `edge_weight_mut` publishes an edge update
/// whether or not a property lands, which is why the SET and REMOVE gates
/// cannot be folded inside it.
#[test]
fn a_refused_write_captures_nothing() {
    for (query, constrain) in [
        (
            "MATCH (a:Person {person_id: 1}), (b:Person {person_id: 2}) \
             CREATE (a)-[:KNOWS {since: 'yesterday'}]->(b)",
            true,
        ),
        ("MATCH ()-[r:KNOWS]->() SET r.since = 'yesterday'", true),
        ("MATCH ()-[r:KNOWS]->() REMOVE r.since", false),
    ] {
        let mut graph = if constrain {
            typed_graph()
        } else {
            required_graph()
        };
        cdc::enable(&mut graph, None).expect("enable capture");
        let from = cdc::status(&graph).expect("enabled").current;

        let error = run_err(&mut graph, query);
        assert!(!error.is_empty());
        cdc::drain_at_commit(&mut graph);

        let published = cdc::read(&graph, from, None).expect("enabled");
        assert!(
            published.is_empty(),
            "`{query}` was refused but published {published:?}"
        );
    }
}

/// The bulk loader's pre-pass owes the same contract, and it runs ahead of
/// Pass A's per-row title writes — which capture immediately.
#[test]
fn a_refused_bulk_frame_captures_nothing() {
    use crate::datatypes::DataFrame;
    use crate::graph::mutation::maintain::add_connections;

    let mut graph = required_graph();
    cdc::enable(&mut graph, None).expect("enable capture");
    let from = cdc::status(&graph).expect("enabled").current;

    let frame = DataFrame::from_cypher_rows(
        vec!["s".to_string(), "t".to_string()],
        vec![vec![Value::Int64(1), Value::Int64(2)]],
    )
    .unwrap();
    let error = add_connections(
        &mut graph,
        frame,
        "KNOWS".to_string(),
        "Person".to_string(),
        "s".to_string(),
        "Person".to_string(),
        "t".to_string(),
        None,
        None,
        None,
    )
    .expect_err("the frame supplies no `since` for a new pair");
    assert!(error.contains("'since'"), "{error}");
    cdc::drain_at_commit(&mut graph);

    let published = cdc::read(&graph, from, None).expect("enabled");
    assert!(
        published.is_empty(),
        "a refused frame published {published:?}"
    );
}
