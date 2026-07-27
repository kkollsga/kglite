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
use std::sync::Arc;

use crate::datatypes::Value;
use crate::graph::schema::{DirGraph, PropertyStorage};
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use crate::graph::storage::GraphRead;

// ─────────────────────────────────────────────────────────────────────
// Fingerprint
// ─────────────────────────────────────────────────────────────────────

/// Everything about a graph that a rollback must restore exactly.
/// Properties as `(key, value)` text pairs, sorted — comparable regardless of
/// the underlying property-storage representation.
type PropPairs = Vec<(String, String)>;

/// `(slot, node_type, id, title, sorted properties, sorted labels)`, keyed by
/// petgraph slot so a node that comes back on a different slot is a failure.
type NodeFingerprint = (usize, String, String, String, PropPairs, Vec<String>);

/// `(slot, src slot, tgt slot, conn type, sorted properties)`.
type EdgeFingerprint = (usize, usize, usize, String, PropPairs);

/// One master column store's rows: `(row id, id, title, sorted properties)`.
///
/// Read from `DirGraph::column_stores` — the *master* handle — not from a
/// node's own `Arc` clone of it. The two can diverge (`Arc::make_mut` forks
/// them on every columnar write), and only the node-side half shows up in
/// `NodeFingerprint`. A rollback that restored the nodes but left the master
/// carrying the failed statement's values would look clean node-side and then
/// resurrect those values on the next `SET` or `save()`.
type MasterRows = Vec<(u32, String, String, PropPairs)>;

#[derive(Debug, PartialEq, Eq)]
struct Fingerprint {
    version: u64,
    node_count: usize,
    edge_count: usize,
    nodes: Vec<NodeFingerprint>,
    edges: Vec<EdgeFingerprint>,
    /// `type_indices` in bucket order — the scan order of `MATCH (n:T)`.
    type_indices: Vec<(String, Vec<usize>)>,
    /// `secondary_label_index` in bucket order.
    secondary_labels: Vec<(String, Vec<usize>)>,
    has_secondary_labels: bool,
    /// Schema growth a failed statement must not leave behind.
    node_type_metadata: Vec<(String, PropPairs)>,
    connection_type_metadata: Vec<String>,
    /// `(node_type, id)` → slot, read through the id index so a stale or
    /// unrebuilt index shows up as a mismatch.
    id_lookup: Vec<(String, String, usize)>,
    /// Master column stores per node type. Empty on a non-columnar graph.
    column_masters: Vec<(String, MasterRows)>,
    /// Per columnar node slot: is its own `Arc<ColumnStore>` handle still the
    /// same allocation as its type's master? `execute_set` writes the master
    /// through `Arc::make_mut` — which forks it away from every node holding a
    /// handle — and then re-points all of them at the fork. A rollback has to
    /// put both halves back *together*: leaving the nodes on the pre-statement
    /// store while the master keeps the fork (or the reverse) is the failure
    /// mode this pins, and it is invisible to a values-only comparison.
    columnar_handles: Vec<(usize, bool)>,
    /// Every user-index bucket as `(index, value, members in bucket order)`.
    /// Order matters: `lookup_by_index` hands the bucket `Vec` straight to the
    /// matcher, so bucket order is the row order an indexed `MATCH` without
    /// `ORDER BY` returns. A rollback that restored membership but appended
    /// the node instead of putting it back where it was would be a visible
    /// reordering, and this is what catches it.
    user_indexes: Vec<(String, String, Vec<usize>)>,
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

    let mut column_masters: Vec<(String, MasterRows)> = graph
        .column_stores
        .iter()
        .map(|(node_type, store)| {
            let rows: MasterRows = (0..store.row_count())
                .map(|row| {
                    let mut props: PropPairs = store
                        .row_properties(row)
                        .into_iter()
                        .map(|(key, value)| {
                            (
                                graph.interner.resolve(key).to_string(),
                                format!("{value:?}"),
                            )
                        })
                        .collect();
                    props.sort();
                    (
                        row,
                        format!("{:?}", store.get_id(row)),
                        format!("{:?}", store.get_title(row)),
                        props,
                    )
                })
                .collect();
            (node_type.clone(), rows)
        })
        .collect();
    column_masters.sort();

    let mut columnar_handles: Vec<(usize, bool)> = Vec::new();
    for idx in graph.graph.node_indices().collect::<Vec<_>>() {
        let Some(node) = graph.graph.node_weight(idx) else {
            continue;
        };
        let PropertyStorage::Columnar { store, .. } = &node.properties else {
            continue;
        };
        let node_type = node.node_type_str(&graph.interner);
        let matches_master = graph
            .column_stores
            .get(node_type)
            .is_some_and(|master| Arc::ptr_eq(store, master));
        columnar_handles.push((idx.index(), matches_master));
    }
    columnar_handles.sort();

    let mut user_indexes: Vec<(String, String, Vec<usize>)> = Vec::new();
    for ((node_type, property), value_map) in &graph.property_indices {
        for (value, members) in value_map {
            user_indexes.push((
                format!("property {node_type}.{property}"),
                format!("{value:?}"),
                members.iter().map(|idx| idx.index()).collect(),
            ));
        }
    }
    for ((node_type, property), btree) in &graph.range_indices {
        for (value, members) in btree {
            user_indexes.push((
                format!("range {node_type}.{property}"),
                format!("{value:?}"),
                members.iter().map(|idx| idx.index()).collect(),
            ));
        }
    }
    for ((node_type, properties), comp_map) in &graph.composite_indices {
        for (value, members) in comp_map {
            user_indexes.push((
                format!("composite {node_type}.{}", properties.join("+")),
                format!("{value:?}"),
                members.iter().map(|idx| idx.index()).collect(),
            ));
        }
    }
    user_indexes.sort();

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
        column_masters,
        columnar_handles,
        user_indexes,
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

/// `seeded()` after `enable_columnar()` — the shape every graph takes the
/// moment it is saved, and keeps for the rest of its life. `save()` calls
/// `enable_columnar` (`io/file.rs`), and only the explicit `disable_columnar`
/// ever empties `column_stores` again, so "has been saved once" is a permanent
/// property of a live graph and every mutation after it runs in this shape.
///
/// The preconditions are asserted, not assumed. If `enable_columnar` ever
/// stopped installing master stores, or stopped re-pointing nodes at them, the
/// columnar arms below would quietly become a second run of the plain fixture
/// and prove nothing.
fn seeded_columnar() -> DirGraph {
    let mut graph = seeded();
    graph.enable_columnar();
    assert!(
        !graph.column_stores.is_empty(),
        "the fixture must own master column stores, or the columnar arms are vacuous"
    );
    let columnar_nodes = graph
        .graph
        .node_indices()
        .filter(|idx| {
            matches!(
                graph.graph.node_weight(*idx).map(|node| &node.properties),
                Some(PropertyStorage::Columnar { .. })
            )
        })
        .count();
    assert_eq!(
        columnar_nodes,
        graph.graph.node_count(),
        "every seeded node must read its properties through a column store"
    );
    graph
}

/// `seeded()` plus one index of each user-created family over `Item`.
///
/// The indexed shape is not exotic: an application indexes the property it
/// looks rows up by, and cannot opt out to get a faster write — dropping the
/// index turns the lookup into a label scan. So this is the configuration a
/// primary store actually runs in, and every shape has to hold in it.
///
/// `qty` is indexed twice on purpose (equality *and* range), because
/// `CREATE RANGE INDEX` in Cypher installs both and the two are maintained by
/// separate code.
fn seeded_indexed() -> DirGraph {
    let mut graph = seeded();
    graph.create_index("Item", "name");
    graph.create_index("Item", "qty");
    graph.create_range_index("Item", "qty");
    graph.create_composite_index("Item", &["name", "qty"]);
    assert!(
        !graph.property_indices.is_empty()
            && !graph.range_indices.is_empty()
            && !graph.composite_indices.is_empty(),
        "all three index families must be live, or the indexed arms are vacuous"
    );
    assert!(
        graph
            .property_indices
            .values()
            .any(|value_map| value_map.values().any(|members| !members.is_empty())),
        "the indexes must have been populated from the seeded nodes"
    );
    graph
}

// ─────────────────────────────────────────────────────────────────────
// Per-shape rollback fidelity
// ─────────────────────────────────────────────────────────────────────

/// Every mid-statement-failure shape, run against every graph configuration.
///
/// One entry generates one module holding one test per fixture, so a failure
/// names both the shape and the configuration it failed in
/// (`create_nodes::columnar`).
///
/// The extra arms exist because the plain fixture cannot detect the bug class
/// this file is for. `seeded()` is never saved and never indexed, so it can
/// never trip a `journal_covers` veto — it takes the journal path
/// unconditionally, and is therefore structurally incapable of noticing a
/// journal path that is wrong for a graph that *has* been saved or indexed.
/// Every shape must hold in every configuration a real application graph can
/// be in.
macro_rules! rollback_shapes {
    ($($(#[$doc:meta])* $name:ident: $query:expr, $scope:expr;)*) => {
        $(
            $(#[$doc])*
            mod $name {
                use super::*;

                /// A fresh in-memory graph: no column stores, no user indexes.
                #[test]
                fn plain() {
                    assert_rolls_back(&mut seeded(), $query, $scope);
                }

                /// The saved-graph shape — `enable_columnar` has installed
                /// master column stores that no mutation path ever removes.
                #[test]
                fn columnar() {
                    assert_rolls_back(&mut seeded_columnar(), $query, $scope);
                }

                /// The indexed shape — one user index of each family, whose
                /// buckets the statement's writes maintain incrementally.
                #[test]
                fn indexed() {
                    assert_rolls_back(&mut seeded_indexed(), $query, $scope);
                }
            }
        )*
    };
}

rollback_shapes! {
    create_nodes:
        "CREATE (:Item {id: 100}), (:Item {id: 101, bad: duration({months: 2147483648})})",
        None;

    /// The first pattern (nodes + edge) commits, the second is rejected by the
    /// write whitelist — so the journal must reverse two nodes and an edge.
    create_nodes_and_edges:
        "CREATE (x:Item {id: 200})-[:LINKS {weight: 1}]->(y:Item {id: 201}), \
                (z:Blocked {id: 202})",
        Some(&["Item"]);

    create_with_secondary_labels:
        "CREATE (:Tag:Hot:Fresh {id: 300}), (:Blocked {id: 301})",
        Some(&["Tag"]);

    /// Every Item is updated, then the expression on the last SET item blows
    /// up — a multi-property SET across multiple rows.
    set_properties:
        "MATCH (n:Item) SET n.qty = n.qty + 1, n.name = 'touched', \
                             n.bad = duration({months: 2147483648})",
        None;

    /// One SET clause writing two node types: the Item write commits, then the
    /// Tag write is rejected by the whitelist mid-clause.
    set_on_second_type:
        "MATCH (n:Item), (t:Tag) WHERE n.id = 1 AND t.id = 1 \
         SET n.marker = 'x', t.marker = 'y'",
        Some(&["Item"]);

    set_label:
        "MATCH (t:Tag {id: 2}) SET t:Hot, t.bad = duration({months: 2147483648})",
        None;

    remove_property_and_label:
        "MATCH (t:Tag {id: 1}) REMOVE t.name, t:Hot \
         CREATE (:Blocked {id: 400})",
        Some(&["Tag"]);

    /// Deletes the middle Item — so its slot is a hole in the middle of the
    /// type_indices bucket, and restoring it at the end instead of in place
    /// would fail the fingerprint.
    detach_delete_one:
        "MATCH (n:Item {id: 2}) DETACH DELETE n CREATE (:Blocked {id: 500})",
        Some(&["Item"]);

    detach_delete_all:
        "MATCH (n) DETACH DELETE n CREATE (:Blocked {id: 501})",
        Some(&["Item", "Tag"]);

    delete_labelled_node:
        "MATCH (t:Tag) DETACH DELETE t CREATE (:Blocked {id: 502})",
        Some(&["Tag"]);

    delete_edge:
        "MATCH ()-[r:LINKS]->() DELETE r CREATE (:Blocked {id: 600})",
        Some(&["Item"]);

    merge_create_arm:
        "MERGE (n:Item {id: 700}) ON CREATE SET n.name = 'new' \
         CREATE (:Blocked {id: 701})",
        Some(&["Item"]);

    merge_match_arm:
        "MERGE (n:Item {id: 1}) ON MATCH SET n.name = 'seen' \
         CREATE (:Blocked {id: 702})",
        Some(&["Item"]);

    foreach:
        "FOREACH (i IN [1, 2, 3] | CREATE (:Item {id: 800 + i})) \
         CREATE (:Blocked {id: 804})",
        Some(&["Item"]);

    multi_clause_create_then_set_then_delete:
        "MATCH (n:Item {id: 3}) SET n.qty = 999 \
         CREATE (:Item {id: 900}) \
         CREATE (:Blocked {id: 901})",
        Some(&["Item"]);
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

/// One statement of each mutating shape, run in an order where each leaves the
/// graph as the next expects it.
const ZERO_COPY_QUERIES: &[&str] = &[
    "CREATE (:Item {id: 2000, name: 'x'})",
    "MATCH (n:Item {id: 1}) SET n.qty = 11, n.name = 'renamed'",
    "MATCH (n:Item {id: 2000}) SET n:Featured",
    "MATCH (a:Item {id: 1}), (b:Item {id: 3}) CREATE (a)-[:LINKS {weight: 2}]->(b)",
    "MATCH (n:Item {id: 2000}) DETACH DELETE n",
    "MERGE (n:Item {id: 2001}) ON CREATE SET n.name = 'merged'",
];

/// Assert no statement copies a node, i.e. every one of them took the journal
/// path rather than the clone checkpoint.
///
/// The counter is `BACKEND_CLONE_NODES`, bumped by `impl Clone for
/// GraphBackend` by the number of nodes copied. That is what makes it a real
/// oracle for *which path ran*: the clone checkpoint forks the whole graph and
/// registers the fixture's node count, while the journal checkpoint clones a
/// `DirGraph` whose backend was deliberately emptied first and registers zero.
/// It counts nodes only — a `HashMap` clone bumps nothing — so it says nothing
/// about how much *else* a statement copied.
fn assert_statements_copy_zero_nodes(graph: &mut DirGraph, fixture: &str) {
    use crate::graph::storage::backend::{backend_clone_nodes, reset_backend_clone_count};

    for &query in ZERO_COPY_QUERIES {
        reset_backend_clone_count();
        run(graph, query);
        assert_eq!(
            backend_clone_nodes(),
            0,
            "statement must not copy any node on the {fixture} fixture: {query}"
        );
    }
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
    assert_statements_copy_zero_nodes(&mut seeded(), "plain");
}

/// The same guard on a graph that has been saved.
///
/// This arm is the one that would have caught the columnar veto. A fresh
/// `seeded()` graph has no column stores and no user indexes, so it cannot
/// take the clone path no matter what the gate says — which made the guard
/// above structurally blind to a gate that sends every *real* application
/// graph down the expensive path.
#[test]
fn journalled_statements_copy_zero_nodes_on_a_saved_graph() {
    assert_statements_copy_zero_nodes(&mut seeded_columnar(), "columnar");
}

/// The same guard on a graph with user-created indexes — the other half of
/// the blind spot.
#[test]
fn journalled_statements_copy_zero_nodes_on_an_indexed_graph() {
    assert_statements_copy_zero_nodes(&mut seeded_indexed(), "indexed");
}

/// Rollback is allowed to be the expensive direction, but it must still not
/// reach for a whole-graph copy on the journal path.
fn assert_rollback_copies_zero_nodes(graph: &mut DirGraph, fixture: &str) {
    use crate::graph::storage::backend::{backend_clone_nodes, reset_backend_clone_count};

    reset_backend_clone_count();
    expect_failure(
        graph,
        "MATCH (n:Item) DETACH DELETE n CREATE (:Blocked {id: 1})",
        Some(&["Item"]),
    );
    assert_eq!(
        backend_clone_nodes(),
        0,
        "rollback must not copy any node on the {fixture} fixture"
    );
}

#[test]
fn journalled_rollback_copies_zero_nodes() {
    assert_rollback_copies_zero_nodes(&mut seeded(), "plain");
}

#[test]
fn journalled_rollback_copies_zero_nodes_on_a_saved_graph() {
    assert_rollback_copies_zero_nodes(&mut seeded_columnar(), "columnar");
}

#[test]
fn journalled_rollback_copies_zero_nodes_on_an_indexed_graph() {
    assert_rollback_copies_zero_nodes(&mut seeded_indexed(), "indexed");
}

/// Pins that the `columnar` arms exercise the master side channel rather than
/// quietly falling through to the per-node setter.
///
/// `execute_set`'s columnar fast path only fires for an in-memory `Columnar`
/// node writing a property that is neither `title` nor `name`. If that stopped
/// holding for this fixture — a storage-mode change, a different fallthrough
/// condition — every columnar rollback arm above would still pass, while
/// testing the write path that was never at risk. The veto this file's change
/// removed was aimed squarely at the master write, so "the master write
/// happens here" is a precondition of the whole exercise, not a detail.
#[test]
fn the_columnar_fixture_writes_through_the_master_store() {
    let mut graph = seeded_columnar();
    let before = fingerprint(&mut graph);
    run(&mut graph, "MATCH (n:Item {id: 1}) SET n.qty = 12345");
    let after = fingerprint(&mut graph);

    assert_ne!(
        before.column_masters, after.column_masters,
        "a successful columnar SET must land in the master store; if it does \
         not, the columnar arms are exercising the per-node fallback"
    );
    assert!(
        after
            .columnar_handles
            .iter()
            .all(|(_, matches_master)| *matches_master),
        "the post-write refresh sweep must leave every node's handle on the \
         new master — that sweep is what the journal has to reverse"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Columnar SET cost: O(changes), not O(type)
// ─────────────────────────────────────────────────────────────────────

/// How many `Item` nodes [`wide_columnar`] seeds.
///
/// Large enough that an O(type) capture is unmistakable next to the single
/// node a one-row `SET` actually changes, small enough to stay a unit test.
const WIDE_ITEMS: usize = 200;

/// A saved graph with many nodes of one type — the shape that separates
/// "captures per node changed" from "captures per node of the type".
///
/// The narrow fixtures cannot do this: `seeded_columnar()` holds three
/// `Item`s, so a per-node sweep and a per-change capture differ by two
/// entries and any threshold that catches the difference is indistinguishable
/// from noise.
fn wide_columnar() -> DirGraph {
    let mut graph = DirGraph::new();
    let rows: Vec<String> = (0..WIDE_ITEMS)
        .map(|i| format!("(:Item {{id: {i}, name: 'n{i}', qty: {i}}})"))
        .collect();
    run(&mut graph, &format!("CREATE {}", rows.join(", ")));
    graph.enable_columnar();
    assert!(
        !graph.column_stores.is_empty(),
        "the fixture must own a master column store, or this test is vacuous"
    );
    graph
}

/// A one-row columnar `SET` must journal a pre-image for the node it changed,
/// not for every node of the type.
///
/// This is the guard for the cost regression the post-merge benchmark caught:
/// `MATCH (i:Item {id: …}) SET i.priority = …` on a saved 100k-node graph ran
/// ~1.8× slower than the whole-graph clone it replaced. The mechanism is the
/// end-of-batch handle-refresh sweep in `execute_set`, which re-points every
/// node's `Arc<ColumnStore>` at the forked master. That sweep goes through
/// `node_weight_mut_silent` — silent towards the WAL recorder, but until this
/// guard existed it fell through to the *recorded* `node_weight_mut` on
/// `MemoryGraph`, so a single-property write cloned a `NodeData` per node of
/// the type into the journal.
///
/// Why the existing guards cannot see it: `journalled_statements_copy_zero_nodes`
/// reads `BACKEND_CLONE_NODES`, which counts backend clones only — the journal
/// path deliberately clones no backend, so the counter reads zero whether the
/// journal captured one pre-image or two hundred. The cost lives entirely
/// inside the journal, so the counter has to as well.
#[test]
fn a_columnar_set_journals_one_pre_image_per_changed_node() {
    use crate::graph::storage::undo::{journal_node_pre_images, reset_journal_node_pre_images};

    let mut graph = wide_columnar();
    reset_journal_node_pre_images();
    run(&mut graph, "MATCH (i:Item {id: 7}) SET i.priority = 3");
    let captured = journal_node_pre_images();

    assert!(
        captured <= 2,
        "a one-row columnar SET captured {captured} node pre-images across \
         {WIDE_ITEMS} nodes of the type; it must be O(nodes changed), not \
         O(nodes of the type) — the handle-refresh sweep is being journalled"
    );
}

/// The same statement on a *plain* (never-saved) graph, as the control.
///
/// Pins that the bound above is a property of the columnar path rather than of
/// this fixture's size: if the plain path ever started capturing per type, the
/// columnar assertion alone would not say which layer regressed.
#[test]
fn a_plain_set_journals_one_pre_image_per_changed_node() {
    use crate::graph::storage::undo::{journal_node_pre_images, reset_journal_node_pre_images};

    let mut graph = wide_columnar();
    graph.disable_columnar();
    reset_journal_node_pre_images();
    run(&mut graph, "MATCH (i:Item {id: 7}) SET i.priority = 3");
    let captured = journal_node_pre_images();

    assert!(
        captured <= 2,
        "a one-row SET on a non-columnar graph captured {captured} node \
         pre-images across {WIDE_ITEMS} nodes"
    );
}

/// Why the checkpoint's second `Arc` on the master is *not* what makes the
/// columnar fast path fork the store.
///
/// The natural reading of `Arc::make_mut(master)` forking is that something
/// else holds a handle, and the statement checkpoint's schema shell does hold
/// one. It is not the cause and removing it would not help: `enable_columnar`
/// points every node of the type at the master, so its strong count is
/// `1 + nodes-of-type` before any checkpoint is opened, and the first write of
/// every statement forks regardless. Pinned here because "drop the shell's
/// handle to stop the fork" is a plausible-sounding fix that would trade the
/// rollback guarantee for nothing.
#[test]
fn every_node_shares_the_master_column_store_handle() {
    let graph = wide_columnar();
    let master = graph
        .column_stores
        .get("Item")
        .expect("the fixture installs a master store for Item");

    assert!(
        Arc::strong_count(master) > 1,
        "the master must be shared with the per-node handles; if it were not, \
         the columnar fast path would mutate in place and need no refresh sweep"
    );
    let sharing = graph
        .graph
        .node_indices()
        .filter(
            |idx| match graph.graph.node_weight(*idx).map(|n| &n.properties) {
                Some(PropertyStorage::Columnar { store, .. }) => Arc::ptr_eq(store, master),
                _ => false,
            },
        )
        .count();
    assert_eq!(
        sharing, WIDE_ITEMS,
        "every node of the type must hold its own handle on the master"
    );
}

/// An indexed graph rolls back through the journal, not through a whole-graph
/// clone. The fidelity half is covered by the `indexed` arm of every shape
/// above; what this pins is the *cost* half — that a user index no longer
/// downgrades the checkpoint for the rest of the session.
#[test]
fn indexed_graph_rolls_back_without_copying_the_graph() {
    use crate::graph::storage::backend::{backend_clone_nodes, reset_backend_clone_count};

    let mut graph = seeded_indexed();
    reset_backend_clone_count();
    assert_rolls_back(
        &mut graph,
        "MATCH (n:Item) SET n.name = 'touched', n.bad = duration({months: 2147483648})",
        None,
    );
    assert_eq!(
        backend_clone_nodes(),
        0,
        "an indexed graph must take the journal path, not the clone checkpoint"
    );
}

/// The bucket-order case the position journal exists for.
///
/// `Item` 1 and 3 share a `qty` after the setup write, so that bucket holds
/// two members in a known order. A statement that moves the *first* member out
/// and then fails must put it back at the front — a rollback that merely
/// restored membership would append it, silently reordering the rows an
/// indexed `MATCH` returns.
#[test]
fn rollback_restores_index_bucket_order_not_just_membership() {
    let mut graph = seeded_indexed();
    run(&mut graph, "MATCH (n:Item {id: 3}) SET n.qty = 10");
    let bucket_before = index_bucket(&graph, "qty", Value::Int64(10));
    assert_eq!(
        bucket_before.len(),
        2,
        "the fixture needs a bucket with two members to have an order at all"
    );

    let before = fingerprint(&mut graph);
    let error = expect_failure(
        &mut graph,
        "MATCH (n:Item {id: 1}) SET n.qty = 999 \
         WITH n MATCH (m:Item {id: 2}) SET m.bad = duration({months: 2147483648})",
        None,
    );
    let after = fingerprint(&mut graph);

    assert_eq!(before, after, "statement must roll back.\nerror: {error}");
    assert_eq!(
        index_bucket(&graph, "qty", Value::Int64(10)),
        bucket_before,
        "the evicted member must come back at its original position"
    );
}

/// One `Item` property-index bucket's members, in bucket order.
fn index_bucket(graph: &DirGraph, property: &str, value: Value) -> Vec<usize> {
    graph
        .property_indices
        .get(&("Item".to_string(), property.to_string()))
        .and_then(|value_map| value_map.get(&value))
        .map(|members| members.iter().map(|idx| idx.index()).collect())
        .unwrap_or_default()
}

// ─────────────────────────────────────────────────────────────────────
// Unique-constraint claims across a rollback
// ─────────────────────────────────────────────────────────────────────
//
// `unique_indices` is parked by `swap_data_scale`, so a journal rollback keeps
// the *failed statement's* occupancy map while the data underneath is restored.
// Its undo story is the per-touched-type rebuild in
// `StatementCheckpoint::rollback`. These tests pin both directions of getting
// that wrong:
//
// - a **phantom claim** — a value the failed statement claimed stays occupied,
//   so a later legitimate insert is rejected forever;
// - a **lost claim** — a value the failed statement released stays free, so a
//   real duplicate is admitted on the next write.
//
// The declared tuple is `Item.name`, an explicit non-`id` property, on purpose:
// `declared_unique_tuples` skips `primary_key == "id"`, so a constraint over
// `id` leaves `unique_indices` empty and would make these tests vacuous.

/// One constraint's claimed values as `(value, holding slot)`, sorted.
type UniqueClaims = Vec<(String, usize)>;

/// `(node_type, constraint properties, claims)` per declared constraint.
type UniqueFingerprint = Vec<(String, Vec<String>, UniqueClaims)>;

/// The whole occupancy map, per declared constraint: constraint tuple → the
/// claimed values and the slot holding each. Slot-level so a claim that comes
/// back pointing at the wrong node is a failure, not just a missing one.
fn unique_fingerprint(graph: &DirGraph) -> UniqueFingerprint {
    let mut out: Vec<_> = graph
        .unique_indices
        .iter()
        .map(|((node_type, properties), occupants)| {
            let mut claims: UniqueClaims = occupants
                .iter()
                .map(|(value, idx)| (format!("{value:?}"), idx.index()))
                .collect();
            claims.sort();
            (node_type.clone(), properties.clone(), claims)
        })
        .collect();
    out.sort();
    out
}

/// `seeded()` plus a declared UNIQUE constraint over `Item.name`.
///
/// Asserts the graph still takes the journal path: declaring a unique
/// constraint touches only `unique_indices` / `unique_constraint_keys`, never
/// `property_indices`, so `journal_covers` stays true. If that ever changes,
/// these tests would silently start exercising the clone path and prove
/// nothing — so the precondition is checked, not assumed.
fn seeded_with_unique_name() -> DirGraph {
    let mut graph = seeded();
    run(
        &mut graph,
        "CREATE CONSTRAINT FOR (i:Item) REQUIRE i.name IS UNIQUE",
    );
    assert_eq!(
        graph.unique_indices.len(),
        1,
        "the constraint must be declared and enforcing"
    );
    assert!(
        graph.property_indices.is_empty()
            && graph.composite_indices.is_empty()
            && graph.range_indices.is_empty(),
        "a unique constraint must not create a user index, or these tests \
         would exercise the clone checkpoint instead of the journal"
    );
    graph
}

/// `seeded()` with a UNIQUE constraint over `Item.qty`, then saved.
///
/// `qty` rather than `name` because the columnar fast path deliberately skips
/// `name`/`title` (they fall through to the inline node setter), so a
/// constraint over `name` would exercise the ordinary journalled write and
/// prove nothing about the master side channel. The seeded `qty` values are
/// distinct, so the constraint is satisfiable.
fn seeded_columnar_with_unique_qty() -> DirGraph {
    let mut graph = seeded();
    run(
        &mut graph,
        "CREATE CONSTRAINT FOR (i:Item) REQUIRE i.qty IS UNIQUE",
    );
    graph.enable_columnar();
    assert_eq!(
        graph.unique_indices.len(),
        1,
        "the constraint must be declared and enforcing"
    );
    assert!(
        !graph.column_stores.is_empty(),
        "the graph must be saved, or this is the plain unique fixture again"
    );
    graph
}

/// A unique claim moved by a *columnar* `SET` must come back.
///
/// This is the shape with no `NodeWeight` entry behind it: the value goes into
/// the master column store, so the node's weight never changes and the journal
/// sees only the per-type `ColumnarHandles` entry. Until that entry started
/// reporting the type as stale, the rebuild was reached only by accident —
/// the handle-refresh sweep captured a pre-image for every node of the type,
/// and each of those marked the type stale on the way past. Removing that
/// per-node capture is what made the report explicit, and this test is what
/// says so: without it a failed columnar `SET` leaves the claim it took behind
/// and the claim it released free.
#[test]
fn rollback_restores_claims_moved_by_a_columnar_property_overwrite() {
    let mut graph = seeded_columnar_with_unique_qty();
    let before = unique_fingerprint(&graph);
    assert!(
        !before.is_empty() && !before[0].2.is_empty(),
        "the constraint must hold claims, or this test is vacuous"
    );

    let error = expect_failure(
        &mut graph,
        "MATCH (i:Item {id: 1}) SET i.qty = 999 \
         WITH i MATCH (j:Item {id: 2}) SET j.bad = duration({months: 2147483648})",
        None,
    );

    assert_eq!(
        unique_fingerprint(&graph),
        before,
        "a claim moved through the master column store must move back.\
         \nerror: {error}"
    );
    // The observable half: 10 is claimed again (no lost claim) and 999 is free
    // (no phantom occupant).
    expect_failure(&mut graph, "CREATE (:Item {id: 40, qty: 10})", None);
    run(&mut graph, "CREATE (:Item {id: 41, qty: 999})");
}

/// A statement that claims a new value and *then* fails must not leave the
/// claim behind. Without the rebuild, `'zeta'` stays occupied by a node that
/// no longer exists and every later insert of it is rejected forever.
#[test]
fn rollback_releases_a_claim_the_failed_statement_added() {
    let mut graph = seeded_with_unique_name();
    let before = unique_fingerprint(&graph);

    // First `CREATE` claims 'zeta'; the second collides with 'b' (held by the
    // Item seeded with id 2), so the statement fails after its first write.
    let error = expect_failure(
        &mut graph,
        "CREATE (:Item {id: 10, name: 'zeta'}), (:Item {id: 11, name: 'b'})",
        None,
    );

    assert_eq!(
        unique_fingerprint(&graph),
        before,
        "the rolled-back claim must be gone.\nerror: {error}"
    );
    // The observable half: 'zeta' is insertable, which it would not be if the
    // phantom claim survived.
    run(&mut graph, "CREATE (:Item {id: 13, name: 'zeta'})");
}

/// A statement that *releases* a claim by deleting its holder and then fails
/// must put the claim back. Without the rebuild, `'a'` stays free and a real
/// duplicate is admitted on the next write.
#[test]
fn rollback_restores_a_claim_the_failed_statement_released() {
    let mut graph = seeded_with_unique_name();
    let before = unique_fingerprint(&graph);

    let error = expect_failure(
        &mut graph,
        "MATCH (i:Item {id: 1}) DETACH DELETE i CREATE (:Item {id: 20, name: 'b'})",
        None,
    );

    assert_eq!(
        unique_fingerprint(&graph),
        before,
        "the released claim must be restored, pointing at the restored slot.\
         \nerror: {error}"
    );
    // The observable half: 'a' is claimed again, so a duplicate is refused.
    expect_failure(&mut graph, "CREATE (:Item {id: 21, name: 'a'})", None);
}

/// The `NodeWeight`-only shape: a property overwrite moves a claim from the old
/// value to the new one without touching identity, so it is invisible to
/// `stale_id_indices` and needs `stale_unique_indices` to carry it.
#[test]
fn rollback_restores_claims_moved_by_a_property_overwrite() {
    let mut graph = seeded_with_unique_name();
    let before = unique_fingerprint(&graph);

    let error = expect_failure(
        &mut graph,
        "MATCH (i:Item {id: 1}) SET i.name = 'renamed' \
         WITH i MATCH (j:Item {id: 2}) SET j.bad = duration({months: 2147483648})",
        None,
    );

    assert_eq!(
        unique_fingerprint(&graph),
        before,
        "an overwritten claim must move back.\nerror: {error}"
    );
    // 'a' is claimed again (no lost claim) and 'renamed' is free (no phantom).
    expect_failure(&mut graph, "CREATE (:Item {id: 30, name: 'a'})", None);
    run(&mut graph, "CREATE (:Item {id: 31, name: 'renamed'})");
}
