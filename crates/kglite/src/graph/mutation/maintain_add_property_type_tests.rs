//! What `update_node_properties` (the `add_property` entry point) records in
//! `node_type_metadata` for a batch of values.
//!
//! The metadata is *observed* knowledge — it is what `describe()` shows and
//! what the mismatch message inside this function compares against — so it has
//! to describe the values the batch actually wrote. Typing the whole batch from
//! its first value made it state the opposite for every heterogeneous batch.

use super::*;
use crate::graph::introspection::describe::compute_description;
use crate::graph::introspection::{ConnectionDetail, CypherDetail, FluentDetail};
use crate::graph::languages::cypher::parser::parse_cypher;
use crate::graph::languages::cypher::planner::schema_check::collect_query_warnings;

/// Three `Person` nodes carrying nothing but an id, in load order.
fn three_people() -> (DirGraph, Vec<NodeIndex>) {
    let mut graph = DirGraph::new();
    let df = DataFrame::from_cypher_rows(
        vec!["id".to_string()],
        (1..=3).map(|id| vec![Value::Int64(id)]).collect(),
    )
    .expect("dataframe");
    add_nodes(
        &mut graph,
        df,
        "Person".to_string(),
        "id".to_string(),
        None,
        None,
    )
    .expect("load");
    let mut nodes: Vec<NodeIndex> = graph
        .type_indices
        .get("Person")
        .expect("the loaded type")
        .iter()
        .collect();
    nodes.sort();
    (graph, nodes)
}

/// The type string `add_property` recorded for `Person.score` after writing
/// `values` — one per node, in order.
fn recorded_for(values: Vec<Value>) -> String {
    let (mut graph, nodes) = three_people();
    let batch: Vec<(Option<NodeIndex>, Value)> = nodes
        .into_iter()
        .zip(values)
        .map(|(node, value)| (Some(node), value))
        .collect();
    update_node_properties(&mut graph, &batch, "score").expect("update");
    graph
        .get_node_type_metadata("Person")
        .and_then(|meta| meta.get("score"))
        .cloned()
        .unwrap_or_else(|| panic!("no type recorded for Person.score"))
}

/// The type string a **bulk load** of the same values records — the answer
/// `add_property` has to agree with wherever a column can hold the values as
/// they are.
fn bulk_recorded_for(values: Vec<Value>) -> String {
    let mut graph = DirGraph::new();
    let rows: Vec<Vec<Value>> = values
        .into_iter()
        .enumerate()
        .map(|(i, value)| vec![Value::Int64(i as i64 + 1), value])
        .collect();
    let df = DataFrame::from_cypher_rows(vec!["id".to_string(), "score".to_string()], rows)
        .expect("dataframe");
    add_nodes(
        &mut graph,
        df,
        "Person".to_string(),
        "id".to_string(),
        None,
        None,
    )
    .expect("load");
    graph
        .get_node_type_metadata("Person")
        .and_then(|meta| meta.get("score"))
        .cloned()
        .unwrap_or_else(|| panic!("no type recorded for Person.score"))
}

#[test]
fn an_all_integer_batch_records_int64() {
    assert_eq!(
        recorded_for(vec![Value::Int64(1), Value::Int64(2), Value::Int64(3)]),
        "Int64"
    );
}

#[test]
fn an_all_string_batch_records_string() {
    assert_eq!(
        recorded_for(vec![
            Value::String("a".to_string()),
            Value::String("b".to_string()),
            Value::String("c".to_string()),
        ]),
        "String"
    );
}

/// The bug: two type families in one batch recorded the *first* value's type,
/// so the metadata claimed every score was an integer while two of the three
/// stored values were not.
#[test]
fn a_two_family_batch_records_mixed() {
    assert_eq!(
        recorded_for(vec![
            Value::Int64(1),
            Value::String("two".to_string()),
            Value::Int64(3),
        ]),
        "mixed",
        "a heterogeneous batch has no single type — recording one is the lie"
    );
}

/// Numeric widening is not heterogeneity: a column holds an `Int64` beside a
/// `Float64`, and the bulk path calls that column `Float64`. Both entry points
/// must answer the same, so the twin is the assertion.
#[test]
fn a_numeric_batch_records_what_a_bulk_load_of_the_same_values_records() {
    let values = || vec![Value::Int64(1), Value::Float64(2.5), Value::Int64(3)];
    assert_eq!(bulk_recorded_for(values()), "Float64");
    assert_eq!(recorded_for(values()), bulk_recorded_for(values()));
}

/// The same invariant on a type the old first-value table had no name for at
/// all: it called a batch of booleans `"Unknown"` while a bulk load of the
/// same values called it `"Boolean"`, and the disagreement showed up as a
/// spurious "Type mismatch" error the next time either path touched the
/// property.
#[test]
fn a_boolean_batch_records_what_a_bulk_load_of_the_same_values_records() {
    let values = || {
        vec![
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Boolean(true),
        ]
    };
    assert_eq!(bulk_recorded_for(values()), "Boolean");
    assert_eq!(recorded_for(values()), bulk_recorded_for(values()));
}

/// Nulls carry no type in a column and carry none here either — a batch of
/// them observes nothing, which this path has always spelled `"Unknown"`.
#[test]
fn an_all_null_batch_records_unknown() {
    assert_eq!(
        recorded_for(vec![Value::Null, Value::Null, Value::Null]),
        "Unknown"
    );
}

/// A null among real values is skipped rather than counted as a second type.
#[test]
fn a_null_does_not_make_a_batch_mixed() {
    assert_eq!(
        recorded_for(vec![Value::Int64(1), Value::Null, Value::Int64(3)]),
        "Int64"
    );
}

/// No row was writable, so nothing was observed: the type map is keyed off the
/// validated rows, and an unwritable batch records no metadata at all.
#[test]
fn a_batch_with_no_writable_row_records_nothing() {
    let (mut graph, _) = three_people();
    let before = graph
        .get_node_type_metadata("Person")
        .cloned()
        .unwrap_or_default();
    let _ = update_node_properties(&mut graph, &[(None, Value::Int64(1))], "score");
    let after = graph
        .get_node_type_metadata("Person")
        .cloned()
        .unwrap_or_default();
    assert_eq!(before, after, "an unwritable batch recorded a type");
}

/// The reason `"mixed"` is the right sentinel rather than a guess: it is a
/// name no type-knowledge source recognises, so the plan-time mismatch family
/// stays silent about the property instead of warning on a comparison that can
/// perfectly well be true for some of the rows.
#[test]
fn a_mixed_property_makes_no_comparison_warn() {
    let (mut graph, nodes) = three_people();
    let batch: Vec<(Option<NodeIndex>, Value)> = nodes
        .into_iter()
        .zip([
            Value::Int64(1),
            Value::String("two".to_string()),
            Value::Int64(3),
        ])
        .map(|(node, value)| (Some(node), value))
        .collect();
    update_node_properties(&mut graph, &batch, "score").expect("update");
    assert_eq!(
        graph
            .get_node_type_metadata("Person")
            .and_then(|meta| meta.get("score"))
            .map(String::as_str),
        Some("mixed")
    );

    for query in [
        "MATCH (p:Person) WHERE p.score > 5 RETURN p",
        "MATCH (p:Person) WHERE p.score = 'two' RETURN p",
        "MATCH (p:Person) WHERE p.score STARTS WITH 'tw' RETURN p",
    ] {
        let parsed = parse_cypher(query).expect("parses");
        let messages = collect_query_warnings(&parsed, &graph, &HashMap::new()).into_messages();
        assert!(
            !messages.iter().any(|m| m.contains("score")),
            "{query} warned about a property nothing knows the type of: {messages:?}"
        );
    }
}

/// `describe()` prints this metadata verbatim, so the sentinel has to survive
/// the round trip as text rather than crashing a renderer that expected one of
/// the type names.
#[test]
fn describe_renders_a_mixed_property() {
    let (mut graph, nodes) = three_people();
    let batch: Vec<(Option<NodeIndex>, Value)> = nodes
        .into_iter()
        .zip([
            Value::Int64(1),
            Value::String("two".to_string()),
            Value::Int64(3),
        ])
        .map(|(node, value)| (Some(node), value))
        .collect();
    update_node_properties(&mut graph, &batch, "score").expect("update");

    let xml = compute_description(
        &graph,
        None,
        &ConnectionDetail::Off,
        &CypherDetail::Off,
        &FluentDetail::Off,
        None,
        None,
        None,
    )
    .expect("describe");
    assert!(xml.contains("score"), "the property is missing: {xml}");
    assert!(xml.contains("mixed"), "the recorded type is missing: {xml}");
}
