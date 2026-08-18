//! Bulk-load enforcement of declared property types (`gate_row`).
//!
//! The Cypher paths are covered where they live (`executor::schema_ddl`);
//! this is the third choke point, the one a `add_nodes` DataFrame load takes,
//! which never goes through the Cypher executor at all.

use super::*;
use crate::graph::property_types::DeclaredType;

/// `Person` rows from `(id, age)` pairs, so a row can carry an age of the
/// wrong type.
fn person_rows(rows: Vec<(i64, Value)>) -> DataFrame {
    let rows: Vec<Vec<Value>> = rows
        .into_iter()
        .map(|(id, age)| vec![Value::Int64(id), age])
        .collect();
    DataFrame::from_cypher_rows(vec!["id".to_string(), "age".to_string()], rows).unwrap()
}

fn load(graph: &mut DirGraph, df: DataFrame) -> Result<(), String> {
    add_nodes(
        graph,
        df,
        "Person".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .map(|_| ())
}

#[test]
fn a_bulk_load_is_refused_when_a_row_violates_a_declared_type() {
    let mut graph = DirGraph::new();
    graph
        .create_property_type_constraint("Person", "age", DeclaredType::Integer)
        .unwrap();

    let error = load(
        &mut graph,
        person_rows(vec![
            (1, Value::Int64(30)),
            (2, Value::String("twenty".to_string())),
        ]),
    )
    .expect_err("a string age must not load under an INTEGER constraint");
    assert!(error.contains("'age'"), "got: {error}");
    assert!(error.contains("INTEGER"), "got: {error}");
    assert!(error.contains("STRING"), "got: {error}");

    // The whole load is refused, not half-applied: `add_nodes` gates every row
    // before writing any of them.
    assert_eq!(graph.graph.node_count(), 0, "no row may land");
}

#[test]
fn a_conforming_bulk_load_still_passes() {
    let mut graph = DirGraph::new();
    graph
        .create_property_type_constraint("Person", "age", DeclaredType::Integer)
        .unwrap();
    load(
        &mut graph,
        person_rows(vec![(1, Value::Int64(30)), (2, Value::Int64(25))]),
    )
    .expect("conforming rows must load");
    assert_eq!(graph.graph.node_count(), 2);
}

/// Null is not a type violation — a type constraint is not an existence
/// constraint — so a sparse column still loads.
#[test]
fn null_cells_do_not_block_a_bulk_load() {
    let mut graph = DirGraph::new();
    graph
        .create_property_type_constraint("Person", "age", DeclaredType::Integer)
        .unwrap();
    load(
        &mut graph,
        person_rows(vec![(1, Value::Int64(30)), (2, Value::Null)]),
    )
    .expect("a null cell satisfies every declared type");
    assert_eq!(graph.graph.node_count(), 2);
}

/// The violation reaches the caller as a typed `ConstraintViolation` through
/// the pending slot, not only as a string — that is what makes it a
/// `ConstraintViolationError` for a binding rather than a generic argument
/// error.
#[test]
fn a_bulk_violation_is_parked_as_a_typed_constraint_error() {
    let mut graph = DirGraph::new();
    graph
        .create_property_type_constraint("Person", "age", DeclaredType::Integer)
        .unwrap();
    let error = load(
        &mut graph,
        person_rows(vec![(1, Value::String("twenty".to_string()))]),
    )
    .expect_err("a string age must not load");

    let violation = graph
        .take_constraint_violation_for(&error)
        .expect("the violation must be parked for the message it produced");
    assert_eq!(
        violation.kind,
        crate::graph::constraints::ConstraintKind::PropertyType
    );
}

// ============================================================================
// A refused load leaves nothing behind
// ============================================================================
//
// The gate exists so a violating batch aborts "with the graph untouched — no
// rollback needed and no half-applied load". Observed *metadata* is part of
// the graph a user can see: `describe()` reports it, `.kgl` persists it, and
// the next load compares against it. A refusal that still records the rejected
// column's type makes the next conforming load warn about a schema the user
// never accepted.

/// `Person` rows with an explicit column set, so a load can carry a column
/// whose type disagrees with what is already recorded.
fn rows_with(id: i64, age: Value, city: Option<Value>) -> DataFrame {
    let mut columns = vec!["id".to_string(), "age".to_string()];
    let mut values = vec![Value::Int64(id), age];
    if let Some(city) = city {
        columns.push("city".to_string());
        values.push(city);
    }
    DataFrame::from_cypher_rows(columns, vec![values]).unwrap()
}

fn load_person(graph: &mut DirGraph, df: DataFrame) -> Result<(), String> {
    add_nodes(
        graph,
        df,
        "Person".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .map(|_| ())
}

/// A type's observed property catalogue: `(node_type, [(property, type)])`.
type ObservedMetadata = Vec<(String, Vec<(String, String)>)>;

/// Everything a refused load must not disturb, as one comparable snapshot:
/// the observed property catalogue, the id-field aliases, and the title-field
/// aliases.
type ObservedState = (
    ObservedMetadata,
    Vec<(String, String)>,
    Vec<(String, String)>,
);

/// Everything a refused load must not disturb, as one comparable snapshot.
fn observed_state(graph: &DirGraph) -> ObservedState {
    let mut metadata: ObservedMetadata = graph
        .node_type_metadata
        .iter()
        .map(|(node_type, props)| {
            let mut props: Vec<(String, String)> =
                props.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            props.sort();
            (node_type.clone(), props)
        })
        .collect();
    metadata.sort();
    let mut ids: Vec<(String, String)> = graph
        .id_field_aliases
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    ids.sort();
    let mut titles: Vec<(String, String)> = graph
        .title_field_aliases
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    titles.sort();
    (metadata, ids, titles)
}

/// A load refused by a **property-type** gate must leave observed metadata
/// exactly as it was.
#[test]
fn a_type_refused_load_leaves_observed_metadata_untouched() {
    let mut graph = DirGraph::new();
    load_person(&mut graph, rows_with(1, Value::Int64(30), None)).unwrap();
    graph
        .create_property_type_constraint("Person", "age", DeclaredType::Integer)
        .unwrap();
    let before = observed_state(&graph);

    load_person(
        &mut graph,
        rows_with(2, Value::String("thirty".to_string()), None),
    )
    .expect_err("a string age must not load under an INTEGER constraint");

    assert_eq!(
        observed_state(&graph),
        before,
        "a refused load recorded the rejected column's type as if it had been accepted"
    );
}

/// The same for a **NOT NULL** gate — the defect predates property types, so
/// the fix has to cover the kind that found it.
#[test]
fn a_not_null_refused_load_leaves_observed_metadata_untouched() {
    let mut graph = DirGraph::new();
    load_person(
        &mut graph,
        rows_with(1, Value::Int64(30), Some(Value::String("Oslo".to_string()))),
    )
    .unwrap();
    graph.create_not_null_constraint("Person", "city").unwrap();
    let before = observed_state(&graph);

    // No `city` column at all, and an `age` whose type disagrees with the
    // recorded one: the row is refused for the missing required property, and
    // the disagreeing column must not be recorded on the way out.
    load_person(
        &mut graph,
        rows_with(2, Value::String("thirty".to_string()), None),
    )
    .expect_err("a row with no city must not load under a NOT NULL constraint");

    assert_eq!(
        observed_state(&graph),
        before,
        "a refused load recorded the rejected column's type as if it had been accepted"
    );
}

/// The user-visible consequence: after a refusal, the next *conforming* load
/// must be clean. Before the fix it reported a type mismatch against a schema
/// the refused load had installed.
#[test]
fn a_conforming_load_after_a_refusal_reports_no_errors() {
    let mut graph = DirGraph::new();
    load_person(&mut graph, rows_with(1, Value::Int64(30), None)).unwrap();
    graph
        .create_property_type_constraint("Person", "age", DeclaredType::Integer)
        .unwrap();
    load_person(
        &mut graph,
        rows_with(2, Value::String("thirty".to_string()), None),
    )
    .expect_err("the violating load must be refused");

    let stats = add_nodes(
        &mut graph,
        rows_with(3, Value::Int64(61), None),
        "Person".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .expect("a conforming load must succeed");
    assert!(
        stats.errors.is_empty(),
        "a conforming load after a refusal must not report errors: {:?}",
        stats.errors
    );
}

/// A large frame is flushed to storage in chunks *during* the build loop, so a
/// gate inside that loop aborts with earlier chunks already written. The load
/// must be all-or-nothing regardless of frame size.
#[test]
fn a_large_load_violating_late_creates_no_rows_at_all() {
    let mut graph = DirGraph::new();
    graph
        .create_property_type_constraint("Person", "age", DeclaredType::Integer)
        .unwrap();

    // Comfortably past the chunk threshold, with the offending row late enough
    // that several chunks would have been flushed before it was reached.
    let mut rows: Vec<Vec<Value>> = (0..5_000)
        .map(|i| vec![Value::Int64(i), Value::Int64(i % 90)])
        .collect();
    rows[4_500][1] = Value::String("ancient".to_string());
    let df = DataFrame::from_cypher_rows(vec!["id".to_string(), "age".to_string()], rows).unwrap();

    load(&mut graph, df).expect_err("a violating row must refuse the whole load");
    assert_eq!(
        graph.graph.node_count(),
        0,
        "a refused load must create no rows, however large the frame"
    );
}
