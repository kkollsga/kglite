use super::resolve_noderefs;
use crate::datatypes::values::{NodeValue, PathValue, RelValue};
use crate::datatypes::{PropMap, Value};
use crate::graph::dir_graph::DirGraph;
use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
use crate::graph::storage::GraphWrite;
use std::collections::HashMap;

fn graph() -> DirGraph {
    let mut graph = DirGraph::new();
    execute_mut(
        &mut graph,
        "CREATE (a:Item {id:'a',title:'Alpha',payload:[0]}), (b:Item {id:'b',title:'Beta'}), (a)-[:LINK]->(b)",
        &ExecuteOptions::eager(&HashMap::new()),
    )
    .unwrap();
    graph
}

#[test]
fn shared_read_and_mutation_boundaries_resolve_nested_endpoint_values() {
    let mut graph = graph();
    let params = HashMap::new();
    let options = ExecuteOptions::eager(&params);
    let expected = Value::Map(PropMap::from_pairs(vec![
        ("first".into(), Value::String("Alpha".into())),
        (
            "pair".into(),
            Value::List(vec![
                Value::String("Alpha".into()),
                Value::String("Beta".into()),
            ]),
        ),
    ]));
    let query = "MATCH ()-[r:LINK]->() RETURN {first:startNode(r),pair:[startNode(r),endNode(r)]} AS result";
    assert_eq!(
        execute_read(&graph, query, &options).unwrap().result.rows,
        vec![vec![expected]]
    );
    let held = execute_read(&graph, query, &options).unwrap().result.rows;
    let result = execute_mut(
        &mut graph,
        "MATCH (a:Item {id:'a'})-[r:LINK]->() SET a.title='Changed' RETURN [startNode(r),endNode(r)] AS result",
        &options,
    )
    .unwrap();
    assert_eq!(
        result.result.rows,
        vec![vec![Value::List(vec![
            Value::String("Changed".into()),
            Value::String("Beta".into())
        ])]]
    );
    let json = crate::param::kglite_value_to_json(&held[0][0]);
    assert_eq!(
        json,
        serde_json::json!({"first":"Alpha","pair":["Alpha","Beta"]})
    );
}

#[test]
fn raw_nested_shapes_preserve_entity_identity_and_shared_originals() {
    let graph = graph();
    let properties = PropMap::from_pairs(vec![
        ("id".into(), Value::String("logical-id".into())),
        ("endpoint".into(), Value::NodeRef(1)),
    ]);
    let node = NodeValue {
        id: 9,
        labels: vec!["Label".into()],
        properties: properties.clone(),
    };
    let rel = RelValue {
        id: 7,
        start_id: 9,
        end_id: 10,
        rel_type: "R".into(),
        properties: properties.clone(),
    };
    let original = vec![vec![
        Value::Node(Box::new(node.clone())),
        Value::Relationship(Box::new(rel.clone())),
        Value::Path(Box::new(PathValue {
            nodes: vec![node.clone()],
            rels: vec![rel.clone()],
        })),
        Value::List(vec![Value::Map(properties.clone())]),
        Value::NodeRef(u32::MAX),
    ]];
    let mut actual = original.clone();
    resolve_noderefs(&graph.graph, &mut actual);
    let mut expected_props = properties.clone();
    expected_props.insert("endpoint", Value::String("Beta".into()));
    let expected_node = NodeValue {
        properties: expected_props.clone(),
        ..node
    };
    let expected_rel = RelValue {
        properties: expected_props.clone(),
        ..rel
    };
    assert_eq!(
        actual,
        vec![vec![
            Value::Node(Box::new(expected_node.clone())),
            Value::Relationship(Box::new(expected_rel.clone())),
            Value::Path(Box::new(PathValue {
                nodes: vec![expected_node],
                rels: vec![expected_rel]
            })),
            Value::List(vec![Value::Map(expected_props)]),
            Value::Null,
        ]]
    );
    assert_eq!(properties.get("endpoint"), Some(&Value::NodeRef(1)));
    assert_ne!(actual, original);
    let once = actual.clone();
    resolve_noderefs(&graph.graph, &mut actual);
    assert_eq!(actual, once);
}

#[test]
fn native_title_reference_cycles_are_unresolved_without_hiding_shared_siblings() {
    // Native invariant control: no Python cyclic input is constructed here.
    let mut graph = graph();
    graph.graph.set_node_title(
        petgraph::graph::NodeIndex::new(0),
        Value::List(vec![Value::NodeRef(0), Value::NodeRef(1)]),
    );
    let mut rows = vec![vec![
        Value::NodeRef(0),
        Value::NodeRef(1),
        Value::NodeRef(1),
    ]];
    resolve_noderefs(&graph.graph, &mut rows);
    assert_eq!(
        rows,
        vec![vec![
            Value::List(vec![Value::Null, Value::String("Beta".into())]),
            Value::String("Beta".into()),
            Value::String("Beta".into())
        ]]
    );
}

#[test]
fn lazy_property_output_resolves_references_in_each_consumer_route() {
    let mut graph = graph();
    let key = graph.interner.get_or_intern("payload");
    graph.graph.set_node_property(
        petgraph::graph::NodeIndex::new(0),
        key,
        Value::List(vec![Value::NodeRef(1)]),
    );
    let params = HashMap::new();
    let mut options = ExecuteOptions::eager(&params);
    options.lazy_eligible = true;
    let disabled = crate::graph::languages::cypher::planner::all_pass_names()
        .into_iter()
        .collect();
    options.disabled_passes = Some(&disabled);
    let result = execute_read(
        &graph,
        "MATCH(n:Item) RETURN n.payload AS payload",
        &options,
    )
    .unwrap()
    .result;
    let lazy = result
        .lazy
        .expect("direct property projection must exercise lazy materialisation");
    let expected = vec![
        vec![Value::List(vec![Value::String("Beta".into())])],
        vec![Value::Null],
    ];
    assert_eq!(
        crate::graph::languages::cypher::result::materialise_lazy(&lazy, &graph).unwrap(),
        expected
    );
    assert_eq!(
        crate::graph::languages::cypher::result::materialise_lazy_row(&lazy, &graph, 0).unwrap(),
        expected[0]
    );
    assert_eq!(
        crate::graph::languages::cypher::result::materialise_lazy_range(&lazy, &graph, 0..1)
            .unwrap(),
        expected[..1]
    );
}

#[test]
fn acyclic_title_reference_chain_retains_exact_terminal_value() {
    let mut graph = DirGraph::new();
    execute_mut(
        &mut graph,
        "UNWIND range(0,63) AS i CREATE (:Item {id:i,title:'Terminal'})",
        &ExecuteOptions::eager(&HashMap::new()),
    )
    .unwrap();
    for index in 0..63 {
        graph.graph.set_node_title(
            petgraph::graph::NodeIndex::new(index),
            Value::NodeRef(index as u32 + 1),
        );
    }
    let mut rows = vec![vec![Value::NodeRef(0), Value::NodeRef(32)]];
    resolve_noderefs(&graph.graph, &mut rows);
    assert_eq!(
        rows,
        vec![vec![
            Value::String("Terminal".into()),
            Value::String("Terminal".into())
        ]]
    );
}
