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
    columnar_rows: Vec<(usize, u32)>,
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
        let Some(node) = graph.graph.node_view(idx) else {
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
        .column_stores_by_name()
        .into_iter()
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
            (node_type.to_string(), rows)
        })
        .collect();
    column_masters.sort();

    let mut columnar_rows: Vec<(usize, u32)> = Vec::new();
    for idx in graph.graph.node_indices().collect::<Vec<_>>() {
        // Deliberately the raw `NodeData`: what this fingerprints is the node's
        // *row identity*, not its property values.
        let Some(node) = graph.graph.node_weight(idx) else {
            continue;
        };
        let PropertyStorage::Columnar(row) = &node.properties else {
            continue;
        };
        // A node carries a row id, not a store handle (D1 Phase 3), so the
        // identity worth fingerprinting is which row it points at.
        columnar_rows.push((idx.index(), row.row_id()));
    }
    columnar_rows.sort();

    let mut user_indexes: Vec<(String, String, Vec<usize>)> = Vec::new();
    for ((node_type, property), value_map) in &graph.property_indices {
        for (value, members) in value_map.iter() {
            user_indexes.push((
                format!("property {node_type}.{property}"),
                format!("{value:?}"),
                members.iter().map(|idx| idx.index()).collect(),
            ));
        }
    }
    for ((node_type, property), btree) in &graph.range_indices {
        for (value, members) in btree.iter() {
            user_indexes.push((
                format!("range {node_type}.{property}"),
                format!("{value:?}"),
                members.iter().map(|idx| idx.index()).collect(),
            ));
        }
    }
    for ((node_type, properties), comp_map) in &graph.composite_indices {
        for (value, members) in comp_map.iter() {
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
        columnar_rows,
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
    seed_into(&mut graph);
    graph
}

/// The seeding itself, factored out so a fixture on a *different backend*
/// is provably the same graph rather than a similar one written twice.
///
/// Every arm below compares behaviour across backends, so a drift between
/// two hand-maintained copies of these queries would surface as a backend
/// difference and be believed.
fn seed_into(graph: &mut DirGraph) {
    run(
        graph,
        "CREATE (a:Item {id: 1, name: 'a', qty: 10}), \
                (b:Item {id: 2, name: 'b', qty: 20}), \
                (c:Item {id: 3, name: 'c', qty: 30})",
    );
    run(graph, "CREATE (t:Tag:Hot {id: 1, name: 'urgent'})");
    run(graph, "CREATE (t:Tag:Cold {id: 2, name: 'later'})");
    run(
        graph,
        "MATCH (a:Item {id: 1}), (b:Item {id: 2}) CREATE (a)-[:LINKS {weight: 5}]->(b)",
    );
    run(
        graph,
        "MATCH (b:Item {id: 2}), (c:Item {id: 3}) CREATE (b)-[:LINKS {weight: 7}]->(c)",
    );
    run(
        graph,
        "MATCH (a:Item {id: 1}), (t:Tag {id: 1}) CREATE (a)-[:TAGGED]->(t)",
    );
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
        graph.is_columnar(),
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

/// `seeded()` on the **Mapped** backend — the storage mode a large-graph
/// application actually runs in.
///
/// This fixture exists because the whole file was memory-only until 2026-07-30:
/// `seeded`, `seeded_columnar` and `seeded_indexed` all build a bare
/// `DirGraph`, so nothing here had ever executed the mapped path. Rollback on a
/// mapped graph was entirely unverified — not "verified and slow", unverified.
///
/// Mapped is not a larger-than-RAM backend in the way the name suggests:
/// `MappedGraph.inner` is the same heap `StableDiGraph<NodeData, EdgeData>` as
/// `MemoryGraph.inner`. What `StorageMode::Mapped` changes is the backend
/// variant plus `memory_limit = Some(0)`, which forces the columnar property
/// store to spill to mmap. That distinction is the whole reason the arms below
/// can be written at all.
fn seeded_mapped() -> DirGraph {
    use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};

    let mut graph = new_dir_graph_in_mode(StorageMode::Mapped, None).expect("mapped graph");
    assert!(
        graph.graph.is_mapped(),
        "the mapped fixture must actually be on the Mapped backend, or every \
         arm below is a second run of the plain fixture"
    );
    seed_into(&mut graph);
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
            .any(|value_map| value_map.iter().any(|(_, members)| !members.is_empty())),
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

                /// The **mapped** shape — the storage mode a large-graph
                /// application runs in, and a journal-path configuration
                /// since 2026-07-30. Before that flip Mapped took the clone
                /// checkpoint, so no shape here had ever been rolled back
                /// through the journal on it.
                #[test]
                fn mapped() {
                    assert_rolls_back(&mut seeded_mapped(), $query, $scope);
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

/// The mapped backend now takes the **journal** path too.
///
/// This arm was the inverse until 2026-07-30: it asserted `copied > 0`,
/// pinning the whole-graph clone every mapped statement used to open, with a
/// note to invert rather than delete it when Mapped gained a journal. This is
/// that inversion. `MappedGraph.inner` is the same heap `StableDiGraph` as
/// `MemoryGraph.inner`, so the same capture seam applies; only `Disk`, which
/// has no petgraph and no `NodeIndex` identity for an `UndoEntry` to name,
/// still returns `false` from `supports_undo_journal`.
///
/// Keeping the arm rather than folding it into the list above is deliberate:
/// it is still the only place the *cost* of a mapped statement is measured,
/// and a regression that quietly re-vetoes Mapped would otherwise be invisible
/// again.
#[test]
fn mapped_statements_copy_zero_nodes() {
    let mut graph = seeded_mapped();
    assert!(
        graph.graph.node_count() > 0,
        "fixture must have nodes or the counter proves nothing"
    );
    assert_statements_copy_zero_nodes(&mut graph, "mapped");
}

/// Mapped rollback must be *correct*, whichever path it takes.
///
/// Separate from the cost arm above on purpose. The clone path and the journal
/// path produce identical observable state by construction, which is exactly
/// why the Python parity oracle cannot tell them apart — it compares outputs.
/// So the cost is pinned by a counter and correctness is pinned by the
/// fingerprint, and neither stands in for the other.
///
/// This is the arm that kept passing unchanged when Mapped switched to the
/// journal. If it ever fails, the journal restored less than the clone used
/// to, and the fingerprint says which field.
#[test]
fn mapped_rolls_back_completely() {
    let mut graph = seeded_mapped();
    assert_rolls_back(
        &mut graph,
        "MATCH (n:Item) DETACH DELETE n CREATE (:Blocked {id: 1})",
        Some(&["Item"]),
    );
}

/// The mapped **silent** write path must record nothing in the undo journal.
///
/// [`GraphWrite::node_weight_mut_silent`] has a *trait default* that forwards
/// to the recorded `node_weight_mut`. `MemoryGraph` overrides it to bypass
/// capture; `MappedGraph` relying on the default was harmless only while
/// Mapped had no journal to capture into. The moment it got one, the default
/// makes every silent write clone a `NodeData` pre-image. The sweeps that used
/// to justify this — the columnar handle refresh and `add_nodes`'
/// detach/reattach dance — are gone with D1 Phase 3, but the override still
/// has to hold: `BatchProcessor::reattach_columnar_stores` remains a silent
/// per-node write, it runs only under `is_mapped() || is_disk()`, and no
/// memory-backed test exercises it. That is exactly the O(type)-per-write
/// amplification commit 3bf9ef00 removed from the WAL, re-created inside the
/// journal.
///
/// Nothing existing catches it. The WAL payload guard
/// (`tests/benchmarks/test_bench_wal_payload.py`) measures WAL *bytes*, which
/// this override does not change; every `BACKEND_CLONE_NODES` arm counts
/// backend clones, and the journal path clones no backend by construction. So
/// the assertion has to read the journal itself.
///
/// The recorded arm is the non-vacuity control: without it a test that
/// silently failed to install a journal, or looked at the wrong node, would
/// pass by reading zero from an empty buffer.
#[test]
fn the_mapped_silent_write_path_records_nothing() {
    use crate::graph::storage::GraphWrite;

    let mut graph = seeded_mapped();
    let idx = graph
        .graph
        .node_indices()
        .next()
        .expect("the fixture must have a node");

    // Control: the recorded seam does capture, so the journal is really live.
    graph.graph.begin_undo();
    GraphWrite::node_weight_mut(&mut graph.graph, idx).expect("node is live");
    let recorded = graph
        .graph
        .take_undo()
        .expect("begin_undo must install a journal on a mapped graph")
        .into_replay_order()
        .count();
    assert_eq!(
        recorded, 1,
        "the recorded seam must capture a pre-image, or the silent arm below \
         is comparing against an empty journal for the wrong reason"
    );

    // The claim: the silent seam captures nothing.
    graph.graph.begin_undo();
    GraphWrite::node_weight_mut_silent(&mut graph.graph, idx).expect("node is live");
    let silent = graph
        .graph
        .take_undo()
        .expect("begin_undo must install a journal on a mapped graph")
        .into_replay_order()
        .count();
    assert_eq!(
        silent, 0,
        "the mapped silent write path journalled {silent} entries; it must \
         journal none, or the columnar detach/reattach and handle-refresh \
         sweeps cost one pre-image per node of the type per chunk"
    );
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

/// The rollback half of the mapped switch. `mapped_rolls_back_completely`
/// pins that the restore is faithful; this pins that it got there without a
/// whole-graph copy.
#[test]
fn journalled_rollback_copies_zero_nodes_on_a_mapped_graph() {
    assert_rollback_copies_zero_nodes(&mut seeded_mapped(), "mapped");
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
    assert_eq!(
        before.columnar_rows, after.columnar_rows,
        "a columnar SET must not move any node to a different row — it writes \
         a cell of the store the backend owns, and the node's row identity is \
         exactly what must stay put"
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
    wide_columnar_into(DirGraph::new())
}

/// [`wide_columnar`] on the **mapped** backend.
///
/// Not interchangeable with `seeded_mapped()`: a mapped graph built by Cypher
/// `CREATE` has an *empty* `column_stores` (the mapped bulk-columnar path in
/// `mutation::batch` is reached by `add_nodes`, not by the Cypher executor),
/// so the master-store write path the cost guard below is about never fires on
/// it. `enable_columnar()` — what `save()` calls — is what puts a mapped graph
/// into the shape a real application graph is in, and the preconditions are
/// asserted so the arm cannot go vacuous if that stops being true.
fn wide_columnar_mapped() -> DirGraph {
    use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};

    let graph = new_dir_graph_in_mode(StorageMode::Mapped, None).expect("mapped graph");
    assert!(graph.graph.is_mapped(), "fixture must be on Mapped");
    wide_columnar_into(graph)
}

fn wide_columnar_into(mut graph: DirGraph) -> DirGraph {
    let rows: Vec<String> = (0..WIDE_ITEMS)
        .map(|i| format!("(:Item {{id: {i}, name: 'n{i}', qty: {i}}})"))
        .collect();
    run(&mut graph, &format!("CREATE {}", rows.join(", ")));
    graph.enable_columnar();
    assert!(
        graph.is_columnar(),
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

/// The same bound on the **mapped** backend, which is where it is easiest to
/// lose and hardest to notice.
///
/// `node_weight_mut_silent` has a trait *default* that forwards to the recorded
/// `node_weight_mut`. `MemoryGraph` overrides it, which is what the arm above
/// pins; `MappedGraph` had no reason to until it gained a journal, and adding
/// the journal without the override re-creates the O(type)-per-write cost
/// exactly. Measured, not assumed: with the override removed this captured
/// **200** pre-images for a one-row `SET`, against 0 with it.
///
/// This arm and `the_mapped_silent_write_path_records_nothing` guard the same
/// override from opposite ends — one at the seam, one through the statement
/// that actually reaches it — because the seam has a second caller
/// (`mutation::batch`'s columnar detach/reattach, gated on
/// `is_mapped() || is_disk()`) that no Cypher statement reaches today and so
/// no end-to-end test can cover.
#[test]
fn a_mapped_columnar_set_journals_one_pre_image_per_changed_node() {
    use crate::graph::storage::undo::{journal_node_pre_images, reset_journal_node_pre_images};

    let mut graph = wide_columnar_mapped();
    reset_journal_node_pre_images();
    run(&mut graph, "MATCH (i:Item {id: 7}) SET i.priority = 3");
    let captured = journal_node_pre_images();

    assert!(
        captured <= 2,
        "a one-row columnar SET on a mapped graph captured {captured} node \
         pre-images across {WIDE_ITEMS} nodes of the type; the mapped \
         handle-refresh sweep is being journalled"
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

/// Two columnar writes in one statement must both be visible.
///
/// The first write of a statement forks the master away from the undo
/// journal's pre-image; the second finds it uniquely owned and mutates in
/// place. Both must land, and a read after the statement must see the second
/// value, not the first.
///
/// Before D1 Phase 3 this pinned a different mechanism: the first write forked
/// away from `1 + N` node handles and registered the type for an end-of-clause
/// re-point sweep, and the assertion was that the sweep had run. There is no
/// sweep now — the store is the backend's and both writes land in it directly —
/// but the observable is unchanged, which is the point of keeping the test.
#[test]
fn two_columnar_writes_in_one_statement_both_land() {
    let mut graph = wide_columnar();

    // Locate the node up front: `id` is an inline canonical field, not a
    // column-store property, so it cannot be used to read back through the
    // per-node handle. `qty` is columnar and seeded to the node's index.
    let idx = graph
        .graph
        .node_indices()
        .find(|i| {
            graph
                .graph
                .node_view(*i)
                .and_then(|n| n.get_property_value("qty"))
                .map(|v| v == crate::datatypes::Value::Int64(1))
                .unwrap_or(false)
        })
        .expect("fixture seeds qty = node index");

    // Two SET clauses in one statement, same type and same property. The first
    // forks away from the journal's pre-image; the second finds the fresh
    // allocation uniquely owned and mutates IN PLACE.
    run(
        &mut graph,
        "MATCH (n:Item {id: 1}) SET n.qty = 111 SET n.qty = 222",
    );

    // Read back through the public route. Both writes must be visible.
    let node = graph.graph.node_view(idx).expect("node still present");
    assert_eq!(
        node.get_property_value("qty"),
        Some(crate::datatypes::Value::Int64(222)),
        "both writes must be visible; reading 1 means the second write landed \
         somewhere the read route does not resolve"
    );

    // And between statements the master is uniquely owned again — the journal
    // released its pre-image at commit, so the next statement's first write
    // mutates in place instead of forking.
    let master = graph.column_store("Item").expect("master");
    assert_eq!(
        Arc::strong_count(master),
        1,
        "between statements nothing but the backend may hold the master, or \
         every write still pays a whole-store copy"
    );
}

/// **Replaces `every_node_shares_the_master_column_store_handle`.**
///
/// That test pinned the pre-D1 design: `enable_columnar` pointed every node of
/// a type at the master, so its strong count was `1 + nodes-of-type` and every
/// first-write-of-a-statement forked the whole store. D1 Phase 3 deleted the
/// node-held handle, and this is the inverted assertion: *no* node holds one,
/// and the master is uniquely owned.
///
/// Keeping the coverage rather than the assertion is deliberate — the property
/// this file cares about is what the refcount implies for `Arc::make_mut`, and
/// that has flipped from "always copies" to "copies only under a checkpoint".
#[test]
fn no_node_holds_a_column_store_handle() {
    let graph = wide_columnar();
    let master = graph
        .column_store("Item")
        .expect("the fixture installs a master store for Item");

    assert_eq!(
        Arc::strong_count(master),
        1,
        "the backend must be the only owner of the master; a second handle \
         means something re-introduced a replica, and every columnar write \
         would silently go back to copying the whole store"
    );

    // Non-vacuity: the nodes really are columnar, they just carry row ids.
    let columnar = graph
        .graph
        .node_indices()
        .filter(|idx| {
            matches!(
                graph.graph.node_weight(*idx).map(|n| &n.properties),
                Some(PropertyStorage::Columnar(_))
            )
        })
        .count();
    assert_eq!(
        columnar, WIDE_ITEMS,
        "every node of the type must still be columnar, or the refcount above \
         is 1 because the fixture stopped being saved"
    );
}

/// **Replaces `fork_detection_is_a_no_op_while_nodes_hold_strong_handles`.**
///
/// The reference-count invariant the whole programme turns on, asserted in both
/// directions (D1 Phase 3, plan step 5):
///
/// - **under an open checkpoint** the undo journal holds the pre-statement
///   store, so the count is ≥ 2 and `Arc::make_mut` forks — rollback has
///   something pristine to restore;
/// - **between statements** nothing else holds it, so the count is 1 and a
///   one-row write mutates one row in place.
///
/// The first direction is also a `debug_assert!` at the write site; this pins
/// the second, which no assertion inside the write can see.
#[test]
fn the_master_is_uniquely_owned_between_statements() {
    let mut graph = wide_columnar();

    assert_eq!(
        Arc::strong_count(graph.column_store("Item").expect("master")),
        1,
        "precondition: uniquely owned before any statement"
    );

    run(&mut graph, "MATCH (n:Item {id: 1}) SET n.qty = 111");

    assert_eq!(
        Arc::strong_count(graph.column_store("Item").expect("master")),
        1,
        "a committed statement must release the journal's pre-image, or the \
         next statement forks the whole store again"
    );
    assert_eq!(
        graph
            .graph
            .node_view(
                graph
                    .graph
                    .node_indices()
                    .find(|i| graph.graph.get_node_id(*i)
                        == Some(crate::datatypes::Value::Int64(1)))
                    .expect("node 1")
            )
            .and_then(|n| n.get_property_value("qty")),
        Some(crate::datatypes::Value::Int64(111)),
        "and the write must actually be visible"
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

/// The range-index counterpart of the bucket-order pin above, run **with a
/// fork outstanding** — the configuration the layered range index introduced.
///
/// `range_indices` is *parked* by `swap_data_scale`: the shell restore does not
/// cover it, so a failed statement's range buckets are put back one inverse
/// edit at a time by `rollback::apply` (`BucketAppended` / `BucketRemoved`).
/// Since the index became a level stack, those inverse edits run against an
/// **overlay** level whenever a reader is holding the base — the writer's
/// `get_mut` / `entry_or_default` materialise the merged bucket into the
/// overlay first. This pins both halves: the writer is restored exactly
/// (position included), and the reader that forced the fork never sees either
/// the failed write or its reversal.
#[test]
fn a_rolled_back_statement_restores_the_parked_range_index_under_a_fork() {
    let mut graph = seeded_indexed();
    run(&mut graph, "MATCH (n:Item {id: 3}) SET n.qty = 10");
    let bucket_before = range_bucket(&graph, "qty", Value::Int64(10));
    assert_eq!(
        bucket_before.len(),
        2,
        "the fixture needs a range bucket with two members to have an order at all"
    );

    // The fork: a held reader, which is what makes the writer's next edit
    // land in a fresh level over a shared base.
    let reader = graph.clone();
    let reader_before = range_bucket(&reader, "qty", Value::Int64(10));

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
        range_bucket(&graph, "qty", Value::Int64(10)),
        bucket_before,
        "the evicted member must come back at its original position in the range bucket"
    );
    assert!(
        range_bucket(&graph, "qty", Value::Int64(999)).is_empty(),
        "the failed statement's new range bucket must be gone"
    );
    assert_eq!(
        range_bucket(&reader, "qty", Value::Int64(10)),
        reader_before,
        "the reader must not have seen the write or its reversal"
    );

    // The index still answers ordered range scans after the round trip.
    let scanned = graph
        .lookup_range(
            "Item",
            "qty",
            std::ops::Bound::Unbounded,
            std::ops::Bound::Unbounded,
        )
        .expect("the range index survives the rollback");
    assert_eq!(
        scanned.len(),
        3,
        "every seeded Item must still be reachable through the range index"
    );
}

/// One `Item` **range**-index bucket's members, in bucket order.
fn range_bucket(graph: &DirGraph, property: &str, value: Value) -> Vec<usize> {
    graph
        .range_indices
        .get(&("Item".to_string(), property.to_string()))
        .and_then(|btree| btree.get(&value))
        .map(|members| members.iter().map(|idx| idx.index()).collect())
        .unwrap_or_default()
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
        graph.is_columnar(),
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

// ─────────────────────────────────────────────────────────────────────
// D2 Phase 2 — rollback while a reader holds the base
// ─────────────────────────────────────────────────────────────────────

/// A failed statement run while a reader is holding the graph must leave
/// **both** graphs exactly as they were.
///
/// This is D2 risk R3, and it is the one unforgivable failure mode in the whole
/// programme: every `UndoEntry` is keyed on a `NodeIndex`/`EdgeIndex` and is
/// reversed through `GraphWrite`. On a `Forked` backend that reversal must land
/// in the *overlay*. If any of it reached the shared base instead, the reader's
/// snapshot would silently acquire a rolled-back write — no error, no crash,
/// and no other test in this file would see it, because every other test owns
/// its graph outright.
///
/// The reader side is the golden snapshot: `fingerprint` is deliberately
/// over-specified (petgraph slot identity, inverted-index bucket *order*, schema
/// metadata, master column rows), so *any* base mutation shows up here, not just
/// one that changes a value a query would return.
#[test]
fn a_rollback_while_a_reader_is_held_touches_neither_graph() {
    use crate::graph::handle::make_dir_graph_mut;
    use std::sync::Arc;

    for (name, build) in [
        ("plain", seeded as fn() -> DirGraph),
        ("columnar", seeded_columnar as fn() -> DirGraph),
        ("indexed", seeded_indexed as fn() -> DirGraph),
    ] {
        let mut writer = Arc::new(build());
        let reader = Arc::clone(&writer);

        // Fingerprinting needs `&mut`, and the reader is shared — so read
        // through a clone of it. The clone is a copy-on-write overlay over the
        // same base, so its content *is* the reader's content.
        let reader_before = fingerprint(&mut (*reader).clone());

        // One `make_dir_graph_mut` for both the fingerprint and the statement:
        // it is what bumps `version`, and `version` is part of the fingerprint
        // *on purpose* (a rolled-back statement must restore it), so calling it
        // twice would fail on a field the statement never touched.
        let writer_before = {
            let graph = make_dir_graph_mut(&mut writer);
            assert!(
                graph.graph.is_forked(),
                "{name}: precondition — a held reader must produce an overlay"
            );
            let before = fingerprint(&mut graph.clone());
            // Fails after its first write: row 1 commits, row 2 violates the
            // write scope.
            expect_failure(
                graph,
                "CREATE (:Item {id: 4000, name: 'first'}), (:Blocked {id: 4001, name: 'second'})",
                Some(&["Item"]),
            );
            before
        };

        assert_eq!(
            fingerprint(&mut (*reader).clone()),
            reader_before,
            "{name}: the reader's graph must be untouched by a write it never \
             asked for — a difference here means the undo journal reversed into \
             the shared base instead of the overlay (D2 R3)"
        );
        assert_eq!(
            fingerprint(&mut (*writer).clone()),
            writer_before,
            "{name}: the writer's failed statement must roll back exactly, \
             overlay or not"
        );
    }
}

/// The forked backend must take the **journal** path, not the clone checkpoint
/// — and the one write it cannot express must cost exactly one copy, not one
/// per statement.
///
/// D2 risk R2: `journal_covers` has exactly one term left,
/// `supports_undo_journal()`. If `Forked` answered `false` there, every
/// statement taken while a view is held would open a
/// `StatementCheckpoint::Clone` — an O(V+E) copy *per statement* instead of the
/// one-off fork this phase removed, i.e. the fix introducing a cliff worse than
/// the defect. The zeros below are what would break.
///
/// The middle assertion is the honest half. An overlay cannot express an
/// adjacency edit (`storage/forked.rs` module doc), so the edge `CREATE`
/// flattens the overlay — one deep copy, the pre-D2 cost. What matters is that
/// it happens **once**: the backend is a plain `Memory` afterwards, so every
/// later statement is back to mutating in place. A per-statement copy here
/// would be the R2 accident wearing a different hat.
#[test]
fn forked_statements_copy_zero_nodes_except_one_flatten() {
    use crate::graph::handle::make_dir_graph_mut;
    use crate::graph::storage::backend::{backend_clone_nodes, reset_backend_clone_count};
    use std::sync::Arc;

    /// Overlay-expressible: node adds and weight writes.
    const OVERLAY_QUERIES: &[&str] = &[
        "CREATE (:Item {id: 2000, name: 'x'})",
        "MATCH (n:Item {id: 1}) SET n.qty = 11, n.name = 'renamed'",
        "MATCH (n:Item {id: 2000}) SET n:Featured",
        "MERGE (n:Item {id: 2001}) ON CREATE SET n.name = 'merged'",
    ];
    /// Rewrites existing nodes' petgraph adjacency, so it flattens first.
    const ADJACENCY_QUERY: &str =
        "MATCH (a:Item {id: 1}), (b:Item {id: 3}) CREATE (a)-[:LINKS {weight: 2}]->(b)";

    let mut writer = Arc::new(seeded());
    let reader = Arc::clone(&writer);
    let fixture_nodes = reader.graph.node_count();

    let graph = make_dir_graph_mut(&mut writer);
    assert!(graph.graph.is_forked(), "precondition: the write forked");

    for &query in OVERLAY_QUERIES {
        reset_backend_clone_count();
        run(graph, query);
        assert_eq!(
            backend_clone_nodes(),
            0,
            "an overlay-expressible statement on a forked backend must copy no node: {query}"
        );
        assert!(
            graph.graph.is_forked(),
            "...and must leave the backend forked: {query}"
        );
    }

    reset_backend_clone_count();
    run(graph, ADJACENCY_QUERY);
    assert_eq!(
        backend_clone_nodes(),
        fixture_nodes,
        "the adjacency write flattens the overlay — exactly one copy of the base"
    );
    assert!(
        !graph.graph.is_forked(),
        "flattening must leave a plain backend, so the copy is paid once"
    );

    reset_backend_clone_count();
    run(graph, "MATCH (n:Item {id: 2000}) DETACH DELETE n");
    run(graph, "CREATE (:Item {id: 2002, name: 'after'})");
    assert_eq!(
        backend_clone_nodes(),
        0,
        "after flattening, later statements mutate in place — one copy per fork, \
         not one per statement"
    );

    // The reader is still holding the pre-fork base and must be untouched by
    // any of it.
    assert_eq!(reader.graph.node_count(), fixture_nodes);
}
