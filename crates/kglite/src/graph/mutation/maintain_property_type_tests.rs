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
