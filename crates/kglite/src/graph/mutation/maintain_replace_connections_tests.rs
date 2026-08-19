use super::*;
use crate::graph::algorithms::Interrupt;
use crate::graph::property_types::DeclaredType;

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

// ============================================================================
// A failed replace leaves the old edges alone
// ============================================================================
//
// `replace_connections` deletes the source's existing edges and then delegates
// to `add_connections`. Every refusal therefore has to happen *before* the
// delete, or the call destroys data and puts nothing back — the mirror image of
// a load that half-applies. The function pre-validates the columns and the
// interner for exactly this reason; these pin the paths that were still
// reachable past the delete.

/// Attempt a replace that is expected to fail, and report whether Doc 1 kept
/// the edges it had.
fn replace_expecting_failure(
    graph: &mut DirGraph,
    pairs: &[(i64, &str)],
    conflict_handling: Option<String>,
) -> String {
    replace_connections(
        graph,
        edges_df(pairs),
        "MENTIONS".to_string(),
        "Doc".to_string(),
        "s".to_string(),
        "Entity".to_string(),
        "t".to_string(),
        None,
        None,
        conflict_handling,
    )
    .expect_err("the replace was expected to fail")
}

/// An unknown conflict-handling mode is rejected by `add_connections` — after
/// `replace_connections` has already deleted the edges it was meant to replace.
#[test]
fn an_invalid_conflict_mode_does_not_destroy_the_existing_edges() {
    let mut graph = doc_entity_graph();
    add_mentions(&mut graph, &[(1, "A"), (1, "B")]);
    assert_eq!(count_edges_of_type(&graph, "Doc", 1, "MENTIONS"), 2);

    let error = replace_expecting_failure(&mut graph, &[(1, "C")], Some("bogus-mode".to_string()));
    assert!(error.contains("conflict handling"), "got: {error}");

    assert_eq!(
        count_edges_of_type(&graph, "Doc", 1, "MENTIONS"),
        2,
        "a rejected replace must leave the edges it was going to replace"
    );
}

/// An edge to a missing endpoint vivifies a stub node, and that stub goes
/// through the node-side constraint gate. A refusal there aborts the add —
/// again, after the delete.
#[test]
fn a_constraint_refused_stub_does_not_destroy_the_existing_edges() {
    let mut graph = doc_entity_graph();
    add_mentions(&mut graph, &[(1, "A"), (1, "B")]);
    // Entity ids are strings, so requiring an integer id means any stub this
    // load would vivify is refused.
    graph
        .create_property_type_constraint("Entity", "id", DeclaredType::Integer)
        .expect_err("existing string ids already violate it");
    // Declare it on a type whose ids *do* satisfy it, and point the edge at a
    // missing endpoint of that type instead.
    graph
        .create_property_type_constraint("Doc", "id", DeclaredType::Integer)
        .expect("Doc ids are integers");

    let error = replace_connections(
        &mut graph,
        DataFrame::from_cypher_rows(
            vec!["s".to_string(), "t".to_string()],
            vec![vec![Value::Int64(1), Value::String("missing-doc".into())]],
        )
        .unwrap(),
        "MENTIONS".to_string(),
        "Doc".to_string(),
        "s".to_string(),
        // Target type is Doc, whose `id` must be an INTEGER — the string
        // endpoint below cannot be vivified.
        "Doc".to_string(),
        "t".to_string(),
        None,
        None,
        None,
    )
    .expect_err("a stub violating the declared id type must refuse the load");
    assert!(error.contains("INTEGER"), "got: {error}");

    assert_eq!(
        count_edges_of_type(&graph, "Doc", 1, "MENTIONS"),
        2,
        "a rejected replace must leave the edges it was going to replace"
    );
}

/// The same refusal reached through a plain `add_connections` (no delete
/// involved): it must abort the call without leaving edges behind. Pass A
/// buffers its edges into the batch and only `batch.execute` writes them, so a
/// refusal in the later vivification pass never commits them.
#[test]
fn a_constraint_refused_stub_leaves_no_edges_from_a_plain_add() {
    let mut graph = doc_entity_graph();
    graph
        .create_property_type_constraint("Doc", "id", DeclaredType::Integer)
        .expect("Doc ids are integers");

    let before = count_edges_of_type(&graph, "Doc", 1, "MENTIONS");
    assert_eq!(before, 0);

    let error = add_connections(
        &mut graph,
        DataFrame::from_cypher_rows(
            vec!["s".to_string(), "t".to_string()],
            vec![
                // A row whose endpoints both exist — buffered by Pass A.
                vec![Value::Int64(1), Value::Int64(2)],
                // A row whose target is missing, so Pass B tries to vivify a
                // stub whose id violates the declared type.
                vec![Value::Int64(1), Value::String("missing".into())],
            ],
        )
        .unwrap(),
        "MENTIONS".to_string(),
        "Doc".to_string(),
        "s".to_string(),
        "Doc".to_string(),
        "t".to_string(),
        None,
        None,
        None,
    )
    .expect_err("a stub violating the declared id type must refuse the load");
    assert!(error.contains("INTEGER"), "got: {error}");

    assert_eq!(
        count_edges_of_type(&graph, "Doc", 1, "MENTIONS"),
        0,
        "a refused add must commit none of its edges, not even the valid rows"
    );
    assert!(
        graph
            .lookup_by_id_readonly("Doc", &Value::String("missing".into()))
            .is_none(),
        "the refused stub must not exist"
    );
}

/// The A3c contract, extended to relationship constraints: a frame the
/// constraint refuses must not cost the caller the edges they already had.
/// `add_connections` gates the same frame, but its gate runs after this
/// function's delete — so the refusal has to be raised here too.
#[test]
fn a_constraint_violating_frame_does_not_destroy_the_existing_edges() {
    let mut graph = doc_entity_graph();
    // Seed with the property the constraint will require, so the declaration
    // itself is clean and the *frame* is the only thing that violates it.
    let seeded = DataFrame::from_cypher_rows(
        vec!["s".to_string(), "t".to_string(), "since".to_string()],
        vec![
            vec![
                Value::Int64(1),
                Value::String("A".into()),
                Value::Int64(2020),
            ],
            vec![
                Value::Int64(1),
                Value::String("B".into()),
                Value::Int64(2021),
            ],
        ],
    )
    .unwrap();
    add_connections(
        &mut graph,
        seeded,
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
    graph
        .create_rel_not_null_constraint("MENTIONS", "since", &Interrupt::default())
        .expect("nothing stored violates it yet");

    let error = replace_connections(
        &mut graph,
        edges_df(&[(1, "C")]),
        "MENTIONS".to_string(),
        "Doc".to_string(),
        "s".to_string(),
        "Entity".to_string(),
        "t".to_string(),
        None,
        None,
        None,
    )
    .expect_err("the frame carries no `since` for the edge it would create");
    assert!(error.contains("'since'"), "got: {error}");
    assert_eq!(
        count_edges_of_type(&graph, "Doc", 1, "MENTIONS"),
        2,
        "the refused replace must leave the original edges in place"
    );
}
