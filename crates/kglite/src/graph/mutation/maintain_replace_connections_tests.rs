use super::*;

fn doc_entity_graph() -> DirGraph {
    let mut g = DirGraph::new();
    let docs = DataFrame::from_cypher_rows(
        vec!["id".to_string()],
        vec![vec![Value::Int64(1)], vec![Value::Int64(2)]],
    )
    .unwrap();
    add_nodes(
        &mut g,
        docs,
        "Doc".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .unwrap();
    let ents = DataFrame::from_cypher_rows(
        vec!["id".to_string()],
        vec![
            vec![Value::String("A".into())],
            vec![Value::String("B".into())],
            vec![Value::String("C".into())],
        ],
    )
    .unwrap();
    add_nodes(
        &mut g,
        ents,
        "Entity".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .unwrap();
    g
}

fn edges_df(pairs: &[(i64, &str)]) -> DataFrame {
    let rows: Vec<Vec<Value>> = pairs
        .iter()
        .map(|(s, t)| vec![Value::Int64(*s), Value::String((*t).into())])
        .collect();
    DataFrame::from_cypher_rows(vec!["s".to_string(), "t".to_string()], rows).unwrap()
}

fn count_edges_of_type(g: &DirGraph, node_type: &str, id: i64, conn: &str) -> usize {
    let idx = g
        .lookup_by_id_readonly(node_type, &Value::Int64(id))
        .unwrap();
    let key = InternedKey::from_str(conn);
    g.graph
        .edges_directed_filtered(idx, petgraph::Direction::Outgoing, Some(key))
        .filter(|e| e.connection_type() == key)
        .count()
}

fn add_mentions(g: &mut DirGraph, pairs: &[(i64, &str)]) {
    add_connections(
        g,
        edges_df(pairs),
        "MENTIONS".to_string(),
        "Doc".to_string(),
        "s".to_string(),
        "Entity".to_string(),
        "t".to_string(),
        None,
        None,
        None,
    )
    .unwrap();
}

fn replace_mentions(g: &mut DirGraph, pairs: &[(i64, &str)]) {
    replace_connections(
        g,
        edges_df(pairs),
        "MENTIONS".to_string(),
        "Doc".to_string(),
        "s".to_string(),
        "Entity".to_string(),
        "t".to_string(),
        None,
        None,
        None,
    )
    .unwrap();
}

/// The defining behaviour: a source's edges of the named type become
/// exactly the supplied set — stale edges are pruned, new ones added.
#[test]
fn replace_sets_exact_edge_set() {
    let mut g = doc_entity_graph();
    add_mentions(&mut g, &[(1, "A"), (1, "B")]);
    assert_eq!(count_edges_of_type(&g, "Doc", 1, "MENTIONS"), 2);

    replace_mentions(&mut g, &[(1, "B"), (1, "C")]);
    assert_eq!(count_edges_of_type(&g, "Doc", 1, "MENTIONS"), 2);
}

/// Only sources present in the input are pruned; other sources keep
/// their edges, and edges of other types from the same source survive.
#[test]
fn replace_is_scoped_to_input_sources_and_type() {
    let mut g = doc_entity_graph();
    add_mentions(&mut g, &[(1, "A"), (2, "A")]);
    add_connections(
        &mut g,
        edges_df(&[(1, "B")]),
        "CITES".to_string(),
        "Doc".to_string(),
        "s".to_string(),
        "Entity".to_string(),
        "t".to_string(),
        None,
        None,
        None,
    )
    .unwrap();

    replace_mentions(&mut g, &[(1, "C")]);

    assert_eq!(count_edges_of_type(&g, "Doc", 1, "MENTIONS"), 1);
    // doc 2 (absent from the input) keeps its MENTIONS edge.
    assert_eq!(count_edges_of_type(&g, "Doc", 2, "MENTIONS"), 1);
    // The CITES edge from doc 1 is a different type — untouched.
    assert_eq!(count_edges_of_type(&g, "Doc", 1, "CITES"), 1);
}

/// Validation runs before any prune — a bad column errors with the
/// graph's existing edges intact.
#[test]
fn replace_validates_before_pruning() {
    let mut g = doc_entity_graph();
    add_mentions(&mut g, &[(1, "A")]);
    let bad = DataFrame::from_cypher_rows(
        vec!["s".to_string(), "wrong".to_string()],
        vec![vec![Value::Int64(1), Value::String("B".into())]],
    )
    .unwrap();
    let err = replace_connections(
        &mut g,
        bad,
        "MENTIONS".to_string(),
        "Doc".to_string(),
        "s".to_string(),
        "Entity".to_string(),
        "t".to_string(),
        None,
        None,
        None,
    );
    assert!(err.is_err());
    // The pre-existing edge must not have been pruned.
    assert_eq!(count_edges_of_type(&g, "Doc", 1, "MENTIONS"), 1);
}
