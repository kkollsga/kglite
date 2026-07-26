//! Fidelity tests for statement rollback.
//!
//! Every test has the same shape: fingerprint the graph, run a statement that
//! fails *after* its first write, fingerprint again, assert equality. The
//! fingerprint is deliberately over-specified — it pins petgraph slot
//! identity, inverted-index bucket *order*, schema metadata, and the version
//! counter, not just node and edge counts — because the interesting failure
//! mode of an undo journal is restoring the right content into the wrong
//! places.
//!
//! Mid-statement failure is forced two ways, both deterministic:
//!
//! - a **write-scope rejection** (`ExecuteOptions::write_scope`), which fires
//!   between two writes of the same statement;
//! - an **expression error** (`duration({months: 2147483648})`), which fires
//!   while evaluating a later pattern's properties.

use std::collections::{HashMap, HashSet};

use crate::datatypes::Value;
use crate::graph::schema::DirGraph;
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use crate::graph::storage::GraphRead;

// ─────────────────────────────────────────────────────────────────────
// Fingerprint
// ─────────────────────────────────────────────────────────────────────

/// Everything about a graph that a rollback must restore exactly.
#[derive(Debug, PartialEq, Eq)]
struct Fingerprint {
    version: u64,
    node_count: usize,
    edge_count: usize,
    /// `(slot, node_type, id, title, sorted properties, sorted labels)` —
    /// keyed by petgraph slot so a node that comes back on a different slot
    /// is a failure.
    nodes: Vec<(
        usize,
        String,
        String,
        String,
        Vec<(String, String)>,
        Vec<String>,
    )>,
    /// `(slot, src slot, tgt slot, conn type, sorted properties)`.
    edges: Vec<(usize, usize, usize, String, Vec<(String, String)>)>,
    /// `type_indices` in bucket order — the scan order of `MATCH (n:T)`.
    type_indices: Vec<(String, Vec<usize>)>,
    /// `secondary_label_index` in bucket order.
    secondary_labels: Vec<(String, Vec<usize>)>,
    has_secondary_labels: bool,
    /// Schema growth a failed statement must not leave behind.
    node_type_metadata: Vec<(String, Vec<(String, String)>)>,
    connection_type_metadata: Vec<String>,
    /// `(node_type, id)` → slot, read through the id index so a stale or
    /// unrebuilt index shows up as a mismatch.
    id_lookup: Vec<(String, String, usize)>,
}

fn sorted_props(props: &HashMap<String, Value>) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = props
        .iter()
        .map(|(k, v)| (k.clone(), format!("{v:?}")))
        .collect();
    out.sort();
    out
}

fn fingerprint(graph: &mut DirGraph) -> Fingerprint {
    let mut nodes = Vec::new();
    for idx in graph.graph.node_indices().collect::<Vec<_>>() {
        let Some(node) = graph.graph.node_weight(idx) else {
            continue;
        };
        let mut labels: Vec<String> = graph
            .node_labels(idx)
            .into_iter()
            .map(|key| graph.interner.resolve(key).to_string())
            .collect();
        labels.sort();
        nodes.push((
            idx.index(),
            node.node_type_str(&graph.interner).to_string(),
            format!("{:?}", node.id()),
            format!("{:?}", node.title()),
            sorted_props(&node.properties_cloned(&graph.interner)),
            labels,
        ));
    }
    nodes.sort();

    let mut edges = Vec::new();
    for eidx in graph.graph.edge_indices().collect::<Vec<_>>() {
        let (Some((src, tgt)), Some(edge)) = (
            graph.graph.edge_endpoints(eidx),
            graph.graph.edge_weight(eidx),
        ) else {
            continue;
        };
        edges.push((
            eidx.index(),
            src.index(),
            tgt.index(),
            edge.connection_type_str(&graph.interner).to_string(),
            sorted_props(&edge.properties_cloned(&graph.interner)),
        ));
    }
    edges.sort();

    let mut type_indices: Vec<(String, Vec<usize>)> = graph
        .type_indices
        .iter()
        .map(|(name, members)| {
            (
                name.to_string(),
                members.to_vec().iter().map(|i| i.index()).collect(),
            )
        })
        .collect();
    type_indices.sort();

    let mut secondary_labels: Vec<(String, Vec<usize>)> = graph
        .secondary_label_index
        .iter()
        .map(|(label, members)| {
            (
                graph.interner.resolve(*label).to_string(),
                members.iter().map(|i| i.index()).collect(),
            )
        })
        .collect();
    secondary_labels.sort();

    let mut node_type_metadata: Vec<(String, Vec<(String, String)>)> = graph
        .node_type_metadata
        .iter()
        .map(|(t, props)| {
            let mut props: Vec<(String, String)> =
                props.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            props.sort();
            (t.clone(), props)
        })
        .collect();
    node_type_metadata.sort();

    let mut connection_type_metadata: Vec<String> =
        graph.connection_type_metadata.keys().cloned().collect();
    connection_type_metadata.sort();

    // Probe the id index for every live node, which forces a lazy rebuild if
    // the rollback invalidated it — a stale index answers with a dead slot.
    let mut id_lookup = Vec::new();
    for entry in &nodes {
        let (slot, node_type, id_debug, ..) = entry;
        if let Some(node) = graph
            .graph
            .node_weight(petgraph::graph::NodeIndex::new(*slot))
        {
            let owned_id = node.id().into_owned();
            if let Some(found) = graph.lookup_by_id(node_type, &owned_id) {
                id_lookup.push((node_type.clone(), id_debug.clone(), found.index()));
            }
        }
    }
    id_lookup.sort();

    Fingerprint {
        version: graph.version,
        node_count: graph.graph.node_count(),
        edge_count: graph.graph.edge_count(),
        nodes,
        edges,
        type_indices,
        secondary_labels,
        has_secondary_labels: graph.has_secondary_labels,
        node_type_metadata,
        connection_type_metadata,
        id_lookup,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Harness
// ─────────────────────────────────────────────────────────────────────

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("setup query failed: {query}: {e}"));
}

/// Run `query` expecting failure, with an optional write whitelist.
fn expect_failure(graph: &mut DirGraph, query: &str, scope: Option<&[&str]>) -> String {
    let params = HashMap::new();
    let mut opts = ExecuteOptions::eager(&params);
    let owned: Option<HashSet<String>> =
        scope.map(|names| names.iter().map(|s| s.to_string()).collect());
    opts.write_scope = owned.as_ref();
    match execute_mut(graph, query, &opts) {
        Ok(_) => panic!("expected {query} to fail mid-statement"),
        Err(error) => error.to_string(),
    }
}

/// Assert `query` fails and leaves the graph byte-for-byte as it was.
fn assert_rolls_back(graph: &mut DirGraph, query: &str, scope: Option<&[&str]>) {
    let before = fingerprint(graph);
    let error = expect_failure(graph, query, scope);
    let after = fingerprint(graph);
    assert_eq!(
        before, after,
        "statement must roll back completely.\nquery: {query}\nerror: {error}"
    );
}

/// A small graph with two node types, edges, secondary labels, and enough
/// property variety that a partial restore shows up in the fingerprint.
fn seeded() -> DirGraph {
    let mut graph = DirGraph::new();
    run(
        &mut graph,
        "CREATE (a:Item {id: 1, name: 'a', qty: 10}), \
                (b:Item {id: 2, name: 'b', qty: 20}), \
                (c:Item {id: 3, name: 'c', qty: 30})",
    );
    run(&mut graph, "CREATE (t:Tag:Hot {id: 1, name: 'urgent'})");
    run(&mut graph, "CREATE (t:Tag:Cold {id: 2, name: 'later'})");
    run(
        &mut graph,
        "MATCH (a:Item {id: 1}), (b:Item {id: 2}) CREATE (a)-[:LINKS {weight: 5}]->(b)",
    );
    run(
        &mut graph,
        "MATCH (b:Item {id: 2}), (c:Item {id: 3}) CREATE (b)-[:LINKS {weight: 7}]->(c)",
    );
    run(
        &mut graph,
        "MATCH (a:Item {id: 1}), (t:Tag {id: 1}) CREATE (a)-[:TAGGED]->(t)",
    );
    graph
}

// ─────────────────────────────────────────────────────────────────────
// Per-shape rollback fidelity
// ─────────────────────────────────────────────────────────────────────

#[test]
fn create_nodes_roll_back() {
    let mut graph = seeded();
    assert_rolls_back(
        &mut graph,
        "CREATE (:Item {id: 100}), (:Item {id: 101, bad: duration({months: 2147483648})})",
        None,
    );
}

#[test]
fn create_nodes_and_edges_roll_back() {
    let mut graph = seeded();
    // The first pattern (nodes + edge) commits, the second is rejected by the
    // write whitelist — so the journal must reverse two nodes and an edge.
    assert_rolls_back(
        &mut graph,
        "CREATE (x:Item {id: 200})-[:LINKS {weight: 1}]->(y:Item {id: 201}), \
                (z:Blocked {id: 202})",
        Some(&["Item"]),
    );
}

#[test]
fn create_with_secondary_labels_rolls_back() {
    let mut graph = seeded();
    assert_rolls_back(
        &mut graph,
        "CREATE (:Tag:Hot:Fresh {id: 300}), (:Blocked {id: 301})",
        Some(&["Tag"]),
    );
}

#[test]
fn set_properties_roll_back() {
    let mut graph = seeded();
    // Every Item is updated, then the expression on the last SET item blows
    // up — a multi-property SET across multiple rows.
    assert_rolls_back(
        &mut graph,
        "MATCH (n:Item) SET n.qty = n.qty + 1, n.name = 'touched', \
                             n.bad = duration({months: 2147483648})",
        None,
    );
}

#[test]
fn set_on_second_type_rolls_back_first() {
    let mut graph = seeded();
    // One SET clause writing two node types: the Item write commits, then the
    // Tag write is rejected by the whitelist mid-clause.
    assert_rolls_back(
        &mut graph,
        "MATCH (n:Item), (t:Tag) WHERE n.id = 1 AND t.id = 1 \
         SET n.marker = 'x', t.marker = 'y'",
        Some(&["Item"]),
    );
}

#[test]
fn set_label_rolls_back() {
    let mut graph = seeded();
    assert_rolls_back(
        &mut graph,
        "MATCH (t:Tag {id: 2}) SET t:Hot, t.bad = duration({months: 2147483648})",
        None,
    );
}

#[test]
fn remove_property_and_label_roll_back() {
    let mut graph = seeded();
    assert_rolls_back(
        &mut graph,
        "MATCH (t:Tag {id: 1}) REMOVE t.name, t:Hot \
         CREATE (:Blocked {id: 400})",
        Some(&["Tag"]),
    );
}

#[test]
fn detach_delete_rolls_back_nodes_edges_and_bucket_order() {
    let mut graph = seeded();
    // Deletes the middle Item — so its slot is a hole in the middle of the
    // type_indices bucket, and restoring it at the end instead of in place
    // would fail the fingerprint.
    assert_rolls_back(
        &mut graph,
        "MATCH (n:Item {id: 2}) DETACH DELETE n CREATE (:Blocked {id: 500})",
        Some(&["Item"]),
    );
}

#[test]
fn detach_delete_all_rolls_back() {
    let mut graph = seeded();
    assert_rolls_back(
        &mut graph,
        "MATCH (n) DETACH DELETE n CREATE (:Blocked {id: 501})",
        Some(&["Item", "Tag"]),
    );
}

#[test]
fn delete_labelled_node_restores_its_labels() {
    let mut graph = seeded();
    assert_rolls_back(
        &mut graph,
        "MATCH (t:Tag) DETACH DELETE t CREATE (:Blocked {id: 502})",
        Some(&["Tag"]),
    );
}

#[test]
fn delete_edge_rolls_back() {
    let mut graph = seeded();
    assert_rolls_back(
        &mut graph,
        "MATCH ()-[r:LINKS]->() DELETE r CREATE (:Blocked {id: 600})",
        Some(&["Item"]),
    );
}

#[test]
fn merge_create_arm_rolls_back() {
    let mut graph = seeded();
    assert_rolls_back(
        &mut graph,
        "MERGE (n:Item {id: 700}) ON CREATE SET n.name = 'new' \
         CREATE (:Blocked {id: 701})",
        Some(&["Item"]),
    );
}

#[test]
fn merge_match_arm_rolls_back() {
    let mut graph = seeded();
    assert_rolls_back(
        &mut graph,
        "MERGE (n:Item {id: 1}) ON MATCH SET n.name = 'seen' \
         CREATE (:Blocked {id: 702})",
        Some(&["Item"]),
    );
}

#[test]
fn foreach_rolls_back() {
    let mut graph = seeded();
    assert_rolls_back(
        &mut graph,
        "FOREACH (i IN [1, 2, 3] | CREATE (:Item {id: 800 + i})) \
         CREATE (:Blocked {id: 804})",
        Some(&["Item"]),
    );
}

#[test]
fn multi_clause_create_then_set_then_delete_rolls_back() {
    let mut graph = seeded();
    assert_rolls_back(
        &mut graph,
        "MATCH (n:Item {id: 3}) SET n.qty = 999 \
         CREATE (:Item {id: 900}) \
         CREATE (:Blocked {id: 901})",
        Some(&["Item"]),
    );
}

// ─────────────────────────────────────────────────────────────────────
// Journal-specific invariants
// ─────────────────────────────────────────────────────────────────────

#[test]
fn rollback_reuses_the_vacated_slots() {
    let mut graph = seeded();
    let slots_before: Vec<usize> = graph
        .graph
        .node_indices()
        .map(|idx| idx.index())
        .collect::<Vec<_>>();
    expect_failure(
        &mut graph,
        "MATCH (n:Item) DETACH DELETE n CREATE (:Blocked {id: 1})",
        Some(&["Item"]),
    );
    let slots_after: Vec<usize> = graph.graph.node_indices().map(|idx| idx.index()).collect();
    assert_eq!(
        slots_before, slots_after,
        "restored nodes must land on the slots they vacated"
    );
}

#[test]
fn successful_statement_leaves_no_journal_installed() {
    let mut graph = seeded();
    run(&mut graph, "CREATE (:Item {id: 1000})");
    assert!(
        graph.graph.take_undo().is_none(),
        "a committed statement must uninstall its journal"
    );
}

#[test]
fn failed_statement_leaves_no_journal_installed() {
    let mut graph = seeded();
    expect_failure(
        &mut graph,
        "CREATE (:Item {id: 1001}), (:Blocked {id: 1002})",
        Some(&["Item"]),
    );
    assert!(
        graph.graph.take_undo().is_none(),
        "a rolled-back statement must uninstall its journal"
    );
}

#[test]
fn a_second_statement_after_a_rollback_still_commits() {
    let mut graph = seeded();
    expect_failure(
        &mut graph,
        "MATCH (n:Item) DETACH DELETE n CREATE (:Blocked {id: 1})",
        Some(&["Item"]),
    );
    // The graph must be fully usable afterwards — including the id index the
    // rollback invalidated and the slots it handed back.
    run(&mut graph, "CREATE (:Item {id: 1100, name: 'after'})");
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let out = execute_mut(&mut graph, "MATCH (n:Item) RETURN count(n) AS c", &opts)
        .expect("read after rollback");
    let rows = out.result.rows;
    assert_eq!(rows.len(), 1);
    assert_eq!(
        format!("{:?}", rows[0][0]),
        format!("{:?}", Value::Int64(4))
    );
}

/// The point of the whole exercise: a mutating statement on the journal path
/// must not copy a single node. This is the regression guard for the
/// O(V+E)-per-write cost the sprint removed — a benchmark would only notice
/// the reintroduction once the graph is large.
///
/// Nodes copied, not clones performed: the checkpoint does clone a `DirGraph`
/// whose backend has been deliberately emptied first (the schema shell), and
/// that O(1) clone is the design, not a regression.
#[test]
fn journalled_statements_copy_zero_nodes() {
    use crate::graph::storage::backend::{backend_clone_nodes, reset_backend_clone_count};

    let mut graph = seeded();
    for query in [
        "CREATE (:Item {id: 2000, name: 'x'})",
        "MATCH (n:Item {id: 1}) SET n.qty = 11, n.name = 'renamed'",
        "MATCH (n:Item {id: 2000}) SET n:Featured",
        "MATCH (a:Item {id: 1}), (b:Item {id: 3}) CREATE (a)-[:LINKS {weight: 2}]->(b)",
        "MATCH (n:Item {id: 2000}) DETACH DELETE n",
        "MERGE (n:Item {id: 2001}) ON CREATE SET n.name = 'merged'",
    ] {
        reset_backend_clone_count();
        run(&mut graph, query);
        assert_eq!(
            backend_clone_nodes(),
            0,
            "statement must not copy any node: {query}"
        );
    }
}

/// Rollback is allowed to be the expensive direction, but it must still not
/// reach for a whole-graph copy on the journal path.
#[test]
fn journalled_rollback_copies_zero_nodes() {
    use crate::graph::storage::backend::{backend_clone_nodes, reset_backend_clone_count};

    let mut graph = seeded();
    reset_backend_clone_count();
    expect_failure(
        &mut graph,
        "MATCH (n:Item) DETACH DELETE n CREATE (:Blocked {id: 1})",
        Some(&["Item"]),
    );
    assert_eq!(backend_clone_nodes(), 0);
}

/// A graph with a user-created property index falls back to the clone
/// checkpoint. The observable contract is the same, and this test exists so
/// the fallback stays exercised rather than becoming dead code.
#[test]
fn indexed_graph_still_rolls_back_via_the_clone_path() {
    let mut graph = seeded();
    graph.create_index("Item", "name");
    assert!(!graph.property_indices.is_empty(), "index must be live");
    assert_rolls_back(
        &mut graph,
        "MATCH (n:Item) SET n.name = 'touched', n.bad = duration({months: 2147483648})",
        None,
    );
}
