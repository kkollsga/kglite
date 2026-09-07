//! Absolute goldens for `keys()` over its three argument families.
//!
//! The differential corpus is blind to these: `keys()` is one shared scalar
//! arm, so an optimised and an unoptimised plan return the same wrong answer.
//! Only checked-in expected values catch a regression here, which is why the
//! map arm returned a silent `null` for as long as it did.

use crate::datatypes::values::Value;
use crate::datatypes::DataFrame;
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::{execute_read, ExecuteOptions};
use std::collections::HashMap;

/// One `Person` with `id`, `title` and one user property.
fn person_graph() -> DirGraph {
    let mut g = DirGraph::new();
    let df = DataFrame::from_cypher_rows(
        vec!["pid".to_string(), "name".to_string(), "age".to_string()],
        vec![vec![
            Value::Int64(1),
            Value::String("Alice".to_string()),
            Value::Int64(30),
        ]],
    )
    .unwrap();
    crate::graph::mutation::maintain::add_nodes(
        &mut g,
        df,
        "Person".to_string(),
        "pid".to_string(),
        Some("name".to_string()),
        None,
    )
    .unwrap();
    g
}

fn first_value(graph: &DirGraph, query: &str, params: HashMap<String, Value>) -> Value {
    let opts = ExecuteOptions {
        params: &params,
        deadline: None,
        max_work_units: None,
        row_limit: None,
        lazy_eligible: false,
        disabled_passes: None,
        embedder: None,
        value_codecs: None,
        cancel: None,
        write_scope: None,
        git_sha: None,
        modified_by: None,
        csv_import: crate::graph::languages::cypher::executor::load_csv::CsvImportPolicy::default(),
        parallel: false,
    };
    let outcome = execute_read(graph, query, &opts).expect("query executes");
    assert_eq!(
        outcome.result.rows.len(),
        1,
        "expected exactly one row from `{query}`"
    );
    outcome.result.rows[0][0].clone()
}

fn strings(value: &Value) -> Vec<String> {
    match value {
        Value::List(items) => items
            .iter()
            .map(|v| match v {
                Value::String(s) => s.clone(),
                other => panic!("expected a string key, got {other:?}"),
            })
            .collect(),
        other => panic!("expected a list, got {other:?}"),
    }
}

#[test]
fn keys_of_a_map_literal_are_its_sorted_key_names() {
    let g = DirGraph::new();
    let value = first_value(&g, "RETURN keys({b: 2, a: 1}) AS k", HashMap::new());
    assert_eq!(strings(&value), vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn keys_of_an_empty_map_is_an_empty_list() {
    let g = DirGraph::new();
    let value = first_value(&g, "RETURN keys({}) AS k", HashMap::new());
    assert_eq!(strings(&value), Vec::<String>::new());
}

#[test]
fn keys_of_a_map_parameter_are_its_key_names() {
    let g = DirGraph::new();
    let mut map = crate::datatypes::prop_map::PropMap::new();
    map.insert("z".to_string(), Value::Int64(1));
    map.insert("y".to_string(), Value::Int64(2));
    let params = HashMap::from([("m".to_string(), Value::Map(map))]);
    let value = first_value(&g, "RETURN keys($m) AS k", params);
    assert_eq!(strings(&value), vec!["y".to_string(), "z".to_string()]);
}

/// The embarrassing case: both halves are ours, and the composition answered
/// `null` while each half answered correctly on its own.
#[test]
fn keys_of_properties_of_a_node_equals_keys_of_that_node() {
    let g = person_graph();
    let via_map = first_value(
        &g,
        "MATCH (n:Person) RETURN keys(properties(n)) AS k",
        HashMap::new(),
    );
    assert_eq!(
        strings(&via_map),
        vec![
            "age".to_string(),
            "id".to_string(),
            "name".to_string(),
            "pid".to_string(),
            "title".to_string(),
            "type".to_string()
        ]
    );
    let via_node = first_value(&g, "MATCH (n:Person) RETURN keys(n) AS k", HashMap::new());
    assert_eq!(strings(&via_map), strings(&via_node));
}

#[test]
fn keys_of_null_is_null() {
    let g = DirGraph::new();
    assert_eq!(
        first_value(&g, "RETURN keys(null) AS k", HashMap::new()),
        Value::Null
    );
}

/// Regression net for the two arms that already worked: the map arm must not
/// shadow a bound node or relationship.
#[test]
fn keys_of_a_bound_node_still_lists_its_properties() {
    let g = person_graph();
    let value = first_value(&g, "MATCH (n:Person) RETURN keys(n) AS k", HashMap::new());
    assert_eq!(
        strings(&value),
        vec![
            "age".to_string(),
            "id".to_string(),
            "name".to_string(),
            "pid".to_string(),
            "title".to_string(),
            "type".to_string()
        ]
    );
}
