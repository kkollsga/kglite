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
