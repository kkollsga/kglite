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

/// `seeded()` after the consolidation pass `save()` runs (`io/file.rs`) — the
/// shape every graph carries for the whole of its life. Nothing empties
/// `column_stores` again, so every mutation after it runs in this shape.
///
/// The preconditions are asserted, not assumed. If `enable_columnar` ever
/// stopped installing master stores, or stopped re-pointing nodes at them, the
/// columnar arms below would quietly become a second run of the plain fixture
/// and prove nothing.
fn seeded_columnar() -> DirGraph {
    let mut graph = seeded();
    graph.enable_columnar();
    assert!(
        graph.column_store_count() > 0,
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

mod cell_fidelity;
mod columnar_cost;
// ─────────────────────────────────────────────────────────────────────
// Shared fixtures used by more than one arm below
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

fn wide_rows_into(mut graph: DirGraph) -> DirGraph {
    let rows: Vec<String> = (0..WIDE_ITEMS)
        .map(|i| format!("(:Item {{id: {i}, name: 'n{i}', qty: {i}}})"))
        .collect();
    run(&mut graph, &format!("CREATE {}", rows.join(", ")));
    graph
}

fn wide_columnar_into(graph: DirGraph) -> DirGraph {
    let mut graph = wide_rows_into(graph);
    graph.enable_columnar();
    assert!(
        graph.column_store_count() > 0,
        "the fixture must own a master column store, or this test is vacuous"
    );
    graph
}

/// A statement's first write, journalled and then rolled back.
///
/// The second clause fails while evaluating its value, which is after the
/// first clause has already written — the shape the whole file is built on.
const FAILS_AFTER_A_COLUMNAR_WRITE: &str = "WITH n MATCH (m:Item {id: 2}) \
     SET m.qty = duration({months: 2147483648})";

/// One `Item`'s property, read through the public node view.
fn item_prop(graph: &DirGraph, id: i64, property: &str) -> Option<Value> {
    let idx = graph
        .graph
        .node_indices()
        .find(|i| graph.graph.get_node_id(*i) == Some(Value::Int64(id)))
        .unwrap_or_else(|| panic!("no Item with id {id}"));
    graph
        .graph
        .node_view(idx)
        .and_then(|n| n.get_property_value(property))
}

mod fidelity;
mod held_reader;
mod journal_invariants;
mod row_undo;
mod store_clone;
mod unique_claims;
