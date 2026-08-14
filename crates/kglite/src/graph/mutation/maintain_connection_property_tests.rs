use super::*;

fn docs(graph: &mut DirGraph, ids: &[i64]) {
    let rows: Vec<Vec<Value>> = ids.iter().map(|i| vec![Value::Int64(*i)]).collect();
    let df = DataFrame::from_cypher_rows(vec!["id".to_string()], rows).unwrap();
    add_nodes(
        graph,
        df,
        "Doc".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .unwrap();
}

/// **Golden — an edge property column that is null in every row must leave
/// no trace.** Not on the edge, not in the connection type's
/// `property_types` (which is serialized into `.kgl` and pinned by the
/// golden-hash test), and not in the persisted interner table.
///
/// This is the contract that constrains any rewrite of `extract_props`:
/// the obvious "resolve the columns once, write a value per column" shape
/// stores a `Null` for the empty column and silently widens every
/// downstream consumer's property set.
///
/// Both passes are covered: row 1 connects two existing nodes (Pass A),
/// row 2 names a target that does not exist yet, so it is deferred,
/// vivified as a stub (Pass B) and replayed through the *same*
/// `extract_props` (Pass C).
#[test]
fn all_null_edge_property_column_stores_nothing() {
    let mut graph = DirGraph::new();
    docs(&mut graph, &[1, 2, 3]);

    let df = DataFrame::from_cypher_rows(
        vec![
            "src".to_string(),
            "tgt".to_string(),
            "weight".to_string(),
            "note".to_string(),
        ],
        vec![
            vec![
                Value::Int64(1),
                Value::Int64(2),
                Value::Int64(7),
                Value::Null,
            ],
            // Target 99 does not exist — deferred, vivified, replayed.
            vec![
                Value::Int64(3),
                Value::Int64(99),
                Value::Int64(9),
                Value::Null,
            ],
        ],
    )
    .unwrap();

    let report = add_connections(
        &mut graph,
        df,
        "LINKS".to_string(),
        "Doc".to_string(),
        "src".to_string(),
        "Doc".to_string(),
        "tgt".to_string(),
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(report.connections_created, 2, "both passes must connect");

    let weight = InternedKey::from_str("weight");
    let note = InternedKey::from_str("note");
    let mut seen = 0;
    for data in graph.graph.edge_weights() {
        seen += 1;
        assert!(
            data.properties.iter().any(|(k, _)| *k == weight),
            "populated column must be stored"
        );
        assert!(
            !data.properties.iter().any(|(k, _)| *k == note),
            "all-null column must not be stored on the edge"
        );
    }
    assert_eq!(seen, 2);

    let meta = graph
        .connection_type_metadata
        .get("LINKS")
        .expect("connection type registered");
    assert!(meta.property_types.contains_key("weight"));
    assert!(
        !meta.property_types.contains_key("note"),
        "all-null column must not enter the connection type's property list"
    );

    assert!(graph.interner.iter().any(|(_, s)| s == "weight"));
    assert!(
        !graph.interner.iter().any(|(_, s)| s == "note"),
        "all-null column must not enter the persisted interner table"
    );
}
