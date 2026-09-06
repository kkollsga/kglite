use super::{execute_mut, execute_read, ExecuteOptions, Session};
use crate::datatypes::{DataFrame, PropMap, Value};
use crate::graph::cdc::{self, CdcEnrichment};
use crate::graph::mutation::{
    add_properties::{add_properties, PropertySpec},
    extend::extend_graph,
    maintain::{add_connections, add_nodes, update_node_properties},
    subgraph::extract_subgraph,
};
use crate::graph::schema::{CurrentSelection, DirGraph, InternedKey};
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

fn execute(graph: &mut DirGraph, query: &str) {
    execute_mut(graph, query, &ExecuteOptions::eager(&HashMap::new())).unwrap();
}

fn fixture() -> DirGraph {
    let mut graph = DirGraph::new();
    execute(
        &mut graph,
        "CREATE (a:Item {id:'a',title:'Alpha'}),(b:Item {id:'b',title:'Beta'}),\
         (a)-[:LINK]->(b),(o:Order {id:'o',title:'Order'})",
    );
    graph
}

fn property(graph: &DirGraph, node: usize, key: &str) -> Option<Value> {
    let _guard = graph.graph.begin_query();
    graph
        .graph
        .get_node_property(NodeIndex::new(node), InternedKey::from_str(key))
}

#[test]
fn cypher_set_and_create_snapshot_scalar_list_map_and_edge_values() {
    let mut graph = fixture();
    execute(
        &mut graph,
        "CREATE CONSTRAINT FOR (n:Item) REQUIRE n.scalar IS :: STRING",
    );
    execute(
        &mut graph,
        "CREATE CONSTRAINT FOR ()-[r:LINK]-() REQUIRE r.endpoint IS :: STRING",
    );
    execute(
        &mut graph,
        "MATCH (a:Item {id:'a'})-[r:LINK]->(b) \
         SET a.scalar=endNode(r),a.nested=[startNode(r),{endpoint:endNode(r)}],\
             r.endpoint=startNode(r) \
         CREATE (c:Copy {id:'c',title:endNode(r),endpoint:endNode(r)})",
    );
    assert_eq!(
        property(&graph, 0, "scalar"),
        Some(Value::String("Beta".into()))
    );
    assert_eq!(
        property(&graph, 0, "nested"),
        Some(Value::List(vec![
            Value::String("Alpha".into()),
            Value::Map(PropMap::from_pairs(vec![(
                "endpoint".into(),
                Value::String("Beta".into()),
            )])),
        ]))
    );
    let edge = graph.graph.edge_indices().next().unwrap();
    assert_eq!(
        graph
            .graph
            .edge_weight(edge)
            .unwrap()
            .properties_cloned(&graph.interner),
        HashMap::from([("endpoint".into(), Value::String("Alpha".into()))])
    );
    assert_eq!(
        property(&graph, 3, "endpoint"),
        Some(Value::String("Beta".into()))
    );
    assert_eq!(
        graph.graph.get_node_title(NodeIndex::new(3)),
        Some(Value::String("Beta".into()))
    );

    execute(&mut graph, "MATCH (b:Item {id:'b'}) SET b.title='Changed'");
    assert_eq!(
        property(&graph, 0, "scalar"),
        Some(Value::String("Beta".into()))
    );
    assert_eq!(
        graph.graph.get_node_title(NodeIndex::new(3)),
        Some(Value::String("Beta".into()))
    );
    let result = execute_read(
        &graph,
        "MATCH ()-[r:LINK]->() RETURN startNode(r)=startNode(r) AS same",
        &ExecuteOptions::eager(&HashMap::new()),
    )
    .unwrap();
    assert_eq!(result.result.rows, vec![vec![Value::Boolean(true)]]);
}

#[test]
fn missing_and_cyclic_native_references_snapshot_to_null() {
    let mut graph = fixture();
    GraphWrite::set_node_title(&mut graph.graph, NodeIndex::new(0), Value::NodeRef(0));
    let err = match execute_mut(
        &mut graph,
        "MATCH (a:Item {id:'a'})-[r:LINK]->() \
         MERGE (:Marker {id:'m',endpoint:startNode(r)})",
        &ExecuteOptions::eager(&HashMap::new()),
    ) {
        Ok(_) => panic!("MERGE unexpectedly accepted a null property"),
        Err(error) => error,
    };
    assert!(err
        .to_string()
        .contains("MERGE cannot use null for property 'endpoint'"));
    assert!(graph.set_node_property(NodeIndex::new(1), "cycle", Value::NodeRef(0)));
    assert!(graph.set_node_property(NodeIndex::new(1), "missing", Value::NodeRef(u32::MAX)));
    assert_eq!(property(&graph, 1, "cycle"), None);
    assert_eq!(property(&graph, 1, "missing"), None);
}

#[test]
fn repeated_merge_matches_its_snapshotted_property() {
    let mut graph = fixture();
    let query = "MATCH (:Item {id:'a'})-[r:LINK]->() \
                 MERGE (:Marker {id:'m',title:endNode(r),endpoint:endNode(r)})";
    execute(&mut graph, query);
    execute(&mut graph, query);
    let result = execute_read(
        &graph,
        "MATCH (m:Marker) RETURN count(m),collect(m.title),collect(m.endpoint)",
        &ExecuteOptions::eager(&HashMap::new()),
    )
    .unwrap();
    assert_eq!(
        result.result.rows,
        vec![vec![
            Value::Int64(1),
            Value::List(vec![Value::String("Beta".into())]),
            Value::List(vec![Value::String("Beta".into())]),
        ]]
    );
}

#[test]
fn normalized_unique_conflict_rolls_back_the_complete_statement() {
    let mut graph = fixture();
    execute(&mut graph, "CREATE (:Item {id:'c',title:'Gamma'})");
    execute(
        &mut graph,
        "MATCH (c:Item {id:'c'}),(b:Item {id:'b'}) CREATE (c)-[:LINK]->(b)",
    );
    execute(
        &mut graph,
        "CREATE CONSTRAINT FOR (n:Item) REQUIRE n.endpoint IS UNIQUE",
    );
    let session = Session::new(graph);
    let mut tx = session.begin();
    let err = match execute_mut(
        tx.working_mut().unwrap(),
        "MATCH (n:Item)-[r:LINK]->() SET n.endpoint=endNode(r)",
        &ExecuteOptions::eager(&HashMap::new()),
    ) {
        Ok(_) => panic!("SET unexpectedly bypassed the unique constraint"),
        Err(error) => error,
    };
    assert!(err.to_string().contains("UNIQUE"));
    drop(tx);
    let snapshot = session.snapshot();
    let a = snapshot
        .lookup_by_id_readonly("Item", &Value::String("a".into()))
        .unwrap();
    let c = snapshot
        .lookup_by_id_readonly("Item", &Value::String("c".into()))
        .unwrap();
    assert_eq!(property(&snapshot, a.index(), "endpoint"), None);
    assert_eq!(property(&snapshot, c.index(), "endpoint"), None);
}

#[test]
fn table_keys_match_the_same_snapshots_that_are_stored() {
    let mut graph = fixture();
    let row = Value::Map(PropMap::from_pairs(vec![
        ("key".into(), Value::NodeRef(1)),
        ("payload".into(), Value::List(vec![Value::NodeRef(0)])),
    ]));
    let params = HashMap::from([("row".into(), row)]);
    let opts = ExecuteOptions::eager(&params);
    let query = "CALL table.upsert({type:'Order',id:'o',property:'rows',key:'key',row:$row}) \
                 YIELD action";
    execute_mut(&mut graph, query, &opts).unwrap();
    execute_mut(&mut graph, query, &opts).unwrap();
    let delete_params = HashMap::from([("value".into(), Value::NodeRef(1))]);
    execute_mut(
        &mut graph,
        "CALL table.delete({type:'Order',id:'o',property:'rows',key:'key',value:$value}) \
         YIELD removed",
        &ExecuteOptions::eager(&delete_params),
    )
    .unwrap();
    execute_mut(&mut graph, query, &opts).unwrap();
    assert_eq!(
        property(&graph, 2, "rows"),
        Some(Value::List(vec![Value::Map(PropMap::from_pairs(vec![
            ("key".into(), Value::String("Beta".into())),
            (
                "payload".into(),
                Value::List(vec![Value::String("Alpha".into())]),
            ),
        ]))]))
    );
}

#[test]
fn native_bulk_routes_snapshot_before_metadata_and_storage() {
    let mut graph = fixture();
    let nodes = DataFrame::from_cypher_rows(
        vec!["id".into(), "title".into(), "payload".into()],
        vec![vec![
            Value::String("bulk".into()),
            Value::String("Bulk".into()),
            Value::Map(PropMap::from_pairs(vec![(
                "nested".into(),
                Value::List(vec![Value::NodeRef(1)]),
            )])),
        ]],
    )
    .unwrap();
    add_nodes(
        &mut graph,
        nodes,
        "Bulk".into(),
        "id".into(),
        Some("title".into()),
        None,
    )
    .unwrap();
    let bulk = graph
        .lookup_by_id("Bulk", &Value::String("bulk".into()))
        .unwrap();
    update_node_properties(&mut graph, &[(Some(bulk), Value::NodeRef(1))], "endpoint").unwrap();

    let edges = DataFrame::from_cypher_rows(
        vec!["source".into(), "target".into(), "payload".into()],
        vec![vec![
            Value::String("a".into()),
            Value::String("b".into()),
            Value::List(vec![Value::NodeRef(1)]),
        ]],
    )
    .unwrap();
    add_connections(
        &mut graph,
        edges,
        "BULK_LINK".into(),
        "Item".into(),
        "source".into(),
        "Item".into(),
        "target".into(),
        None,
        None,
        None,
    )
    .unwrap();
    execute(&mut graph, "MATCH (b:Item {id:'b'}) SET b.title='Later'");

    assert_eq!(
        property(&graph, bulk.index(), "payload"),
        Some(Value::Map(PropMap::from_pairs(vec![(
            "nested".into(),
            Value::List(vec![Value::String("Beta".into())]),
        )])))
    );
    assert_eq!(
        property(&graph, bulk.index(), "endpoint"),
        Some(Value::String("Beta".into()))
    );
    let result = execute_read(
        &graph,
        "MATCH ()-[r:BULK_LINK]->() RETURN r.payload",
        &ExecuteOptions::eager(&HashMap::new()),
    )
    .unwrap();
    assert_eq!(
        result.result.rows,
        vec![vec![Value::List(vec![Value::String("Beta".into())])]]
    );
}

#[test]
fn native_titles_snapshot_separately_from_structural_ids() {
    let mut graph = fixture();
    let structured_id = Value::List(vec![Value::NodeRef(1)]);
    let nodes =
        DataFrame::from_cypher_rows(vec!["id".into()], vec![vec![structured_id.clone()]]).unwrap();
    add_nodes(
        &mut graph,
        nodes,
        "Structured".into(),
        "id".into(),
        None,
        None,
    )
    .unwrap();
    let added = graph
        .lookup_by_id("Structured", &structured_id)
        .expect("the structural id must retain NodeRef identity");
    assert_eq!(graph.graph.get_node_id(added), Some(structured_id.clone()));
    assert_eq!(
        graph.graph.get_node_title(added),
        Some(Value::List(vec![Value::String("Beta".into())]))
    );
    let connection = DataFrame::from_cypher_rows(
        vec!["endpoint".into(), "target".into()],
        vec![vec![structured_id.clone(), Value::String("b".into())]],
    )
    .unwrap();
    add_connections(
        &mut graph,
        connection,
        "SAME_FIELD_TITLE".into(),
        "Structured".into(),
        "endpoint".into(),
        "Item".into(),
        "target".into(),
        Some("endpoint".into()),
        None,
        None,
    )
    .unwrap();
    assert_eq!(graph.graph.get_node_id(added), Some(structured_id.clone()));
    assert_eq!(
        graph.graph.get_node_title(added),
        Some(Value::List(vec![Value::String("Beta".into())]))
    );
    let cross_id = Value::List(vec![Value::NodeRef(0)]);
    let cross_target =
        DataFrame::from_cypher_rows(vec!["id".into()], vec![vec![cross_id.clone()]]).unwrap();
    add_nodes(
        &mut graph,
        cross_target,
        "CrossTarget".into(),
        "id".into(),
        None,
        None,
    )
    .unwrap();
    let cross_connection = DataFrame::from_cypher_rows(
        vec!["source".into(), "target".into()],
        vec![vec![structured_id.clone(), cross_id]],
    )
    .unwrap();
    add_connections(
        &mut graph,
        cross_connection,
        "CROSS_FIELD_TITLE".into(),
        "Structured".into(),
        "source".into(),
        "CrossTarget".into(),
        "target".into(),
        Some("target".into()),
        None,
        None,
    )
    .unwrap();
    assert_eq!(graph.graph.get_node_id(added), Some(structured_id.clone()));
    assert_eq!(
        graph.graph.get_node_title(added),
        Some(Value::List(vec![Value::String("Alpha".into())]))
    );
    execute(
        &mut graph,
        "MATCH (a:Item {id:'a'}) SET a.title='Later Alpha'",
    );
    assert_eq!(
        graph.graph.get_node_title(added),
        Some(Value::List(vec![Value::String("Alpha".into())]))
    );

    execute(&mut graph, "MATCH (b:Item {id:'b'}) SET b.title='Later'");
    assert_eq!(graph.graph.get_node_id(added), Some(structured_id));
    assert_eq!(
        graph.graph.get_node_title(added),
        Some(Value::List(vec![Value::String("Alpha".into())]))
    );
}

#[test]
fn connection_titles_snapshot_and_refuse_before_indexes_or_capture_change() {
    let mut graph = fixture();
    let titles = DataFrame::from_cypher_rows(
        vec![
            "source".into(),
            "target".into(),
            "source_title".into(),
            "target_title".into(),
        ],
        vec![vec![
            Value::String("a".into()),
            Value::String("b".into()),
            Value::List(vec![Value::NodeRef(1)]),
            Value::Map(PropMap::from_pairs(vec![(
                "from".into(),
                Value::NodeRef(0),
            )])),
        ]],
    )
    .unwrap();
    add_connections(
        &mut graph,
        titles,
        "TITLED".into(),
        "Item".into(),
        "source".into(),
        "Item".into(),
        "target".into(),
        Some("source_title".into()),
        Some("target_title".into()),
        None,
    )
    .unwrap();
    assert_eq!(
        graph.graph.get_node_title(NodeIndex::new(0)),
        Some(Value::List(vec![Value::String("Beta".into())]))
    );
    assert_eq!(
        graph.graph.get_node_title(NodeIndex::new(1)),
        Some(Value::Map(PropMap::from_pairs(vec![(
            "from".into(),
            Value::String("Alpha".into()),
        )])))
    );
    execute(
        &mut graph,
        "MATCH (b:Item {id:'b'}) DETACH DELETE b CREATE (:Item {id:'replacement',title:'Later'})",
    );
    assert_eq!(
        graph.graph.get_node_title(NodeIndex::new(0)),
        Some(Value::List(vec![Value::String("Beta".into())]))
    );

    let mut graph = fixture();
    execute(&mut graph, "CREATE (:Item {id:'c',title:['Beta']})");
    execute(
        &mut graph,
        "CREATE CONSTRAINT unique_item_title FOR (n:Item) REQUIRE n.title IS UNIQUE",
    );
    execute(&mut graph, "CREATE RANGE INDEX FOR (n:Item) ON (n.title)");
    cdc::enable(&mut graph, None, CdcEnrichment::Off).unwrap();
    let capture_before = graph.graph.recording().unwrap().ops_len();
    let version_before = graph.version();
    let edges_before = graph.graph.edge_count();
    let conflicting = DataFrame::from_cypher_rows(
        vec!["source".into(), "target".into(), "source_title".into()],
        vec![vec![
            Value::String("a".into()),
            Value::String("b".into()),
            Value::List(vec![Value::NodeRef(1)]),
        ]],
    )
    .unwrap();
    let error = add_connections(
        &mut graph,
        conflicting,
        "REFUSED".into(),
        "Item".into(),
        "source".into(),
        "Item".into(),
        "target".into(),
        Some("source_title".into()),
        None,
        None,
    )
    .expect_err("the normalized title duplicates c.title");
    assert!(error.contains("UNIQUE"));
    assert_eq!(
        graph.graph.get_node_title(NodeIndex::new(0)),
        Some(Value::String("Alpha".into()))
    );
    assert_eq!(graph.graph.edge_count(), edges_before);
    assert_eq!(graph.version(), version_before);
    assert_eq!(graph.graph.recording().unwrap().ops_len(), capture_before);
    assert_eq!(
        graph
            .lookup_by_index("Item", "title", &Value::String("Alpha".into()),)
            .unwrap(),
        vec![NodeIndex::new(0)]
    );
    let valid = DataFrame::from_cypher_rows(
        vec!["source".into(), "target".into(), "source_title".into()],
        vec![vec![
            Value::String("a".into()),
            Value::String("b".into()),
            Value::String("Delta".into()),
        ]],
    )
    .unwrap();
    add_connections(
        &mut graph,
        valid,
        "INDEXED".into(),
        "Item".into(),
        "source".into(),
        "Item".into(),
        "target".into(),
        Some("source_title".into()),
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        graph
            .lookup_by_index("Item", "title", &Value::String("Delta".into()))
            .unwrap(),
        vec![NodeIndex::new(0)]
    );
}

#[test]
fn add_properties_snapshots_raw_source_values_at_its_admission_seam() {
    let mut graph = fixture();
    let raw_key = graph.interner.get_or_intern("raw_endpoint");
    GraphWrite::set_node_property(
        &mut graph.graph,
        NodeIndex::new(0),
        raw_key,
        Value::NodeRef(1),
    );
    let mut selection = CurrentSelection::new();
    selection
        .get_level_mut(0)
        .unwrap()
        .add_selection(None, vec![NodeIndex::new(0)]);
    selection.add_level();
    selection
        .get_level_mut(1)
        .unwrap()
        .add_selection(Some(NodeIndex::new(0)), vec![NodeIndex::new(2)]);
    add_properties(
        &mut graph,
        &selection,
        HashMap::from([(
            "Item".into(),
            PropertySpec::CopyList(vec!["raw_endpoint".into()]),
        )]),
    )
    .unwrap();
    execute(&mut graph, "MATCH (b:Item {id:'b'}) SET b.title='Later'");
    assert_eq!(
        property(&graph, 2, "raw_endpoint"),
        Some(Value::String("Beta".into()))
    );
}

#[test]
fn extend_and_subset_resolve_against_the_source_view() {
    let mut source = fixture();
    GraphWrite::set_node_title(&mut source.graph, NodeIndex::new(0), Value::NodeRef(1));
    let key = source.interner.get_or_intern("endpoint");
    GraphWrite::set_node_property(&mut source.graph, NodeIndex::new(0), key, Value::NodeRef(1));
    assert_eq!(
        source.graph.get_node_title(NodeIndex::new(0)),
        Some(Value::NodeRef(1))
    );

    let mut target = DirGraph::new();
    execute(
        &mut target,
        "CREATE (:Other {id:'x',title:'Wrong0'}),(:Other {id:'y',title:'Wrong1'})",
    );
    extend_graph(&mut target, &source, None).unwrap();
    let copied = target
        .lookup_by_id("Item", &Value::String("a".into()))
        .unwrap();
    assert_eq!(
        target.graph.get_node_property(copied, key),
        Some(Value::String("Beta".into()))
    );
    assert_eq!(
        target.graph.get_node_title(copied),
        Some(Value::String("Beta".into()))
    );

    let mut selection = CurrentSelection::new();
    selection
        .get_level_mut(0)
        .unwrap()
        .add_selection(None, vec![NodeIndex::new(0), NodeIndex::new(1)]);
    let subset = extract_subgraph(&source, &selection).unwrap();
    assert_eq!(
        subset.graph.get_node_property(NodeIndex::new(0), key),
        Some(Value::String("Beta".into()))
    );
    assert_eq!(
        subset.graph.get_node_title(NodeIndex::new(0)),
        Some(Value::String("Beta".into()))
    );
}
