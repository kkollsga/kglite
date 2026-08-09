//! D1 — column-store ownership: divergence coverage and the ownership pins.
//!
//! Programme: `dev-docs/plans/d1-column-store-ownership.md`.
//!
//! # What "divergence" means here
//!
//! A columnar type's `ColumnStore` is reachable through **two** `Arc`s on a
//! memory/mapped graph: `DirGraph.column_stores[type]` (the master) and the
//! handle inside every node's `PropertyStorage::Columnar`. Nothing in the type
//! system stops those from drifting apart — `Arc::make_mut` on either side
//! forks it — and the whole point of D1 is to delete one of them.
//!
//! This module forks them deliberately and then asks every public read surface
//! what it sees. Two classes of assertion live here:
//!
//! 1. **Cross-surface consistency** (`all_public_reads_agree_*`). Whatever a
//!    read resolves to, *every* surface must resolve to the same thing. This
//!    holds today (all of them read the node handle) and must still hold after
//!    Phase 3 (all of them will read the backend's store). It is
//!    phase-independent and is the real gate.
//! 2. **Which replica wins** (`*_today_*`). Pinned as an exact fact with an
//!    inversion instruction, in the style of `handle.rs`'s
//!    `held_reader_forces_a_whole_graph_copy`. Phase 3 flips these; a failure
//!    before then means an unintended ownership change.
//!
//! # Phase 2 — the mutation-proof gate
//!
//! Phase 1 re-routed every caller; Phase 2 makes that irreversible. Two layers:
//!
//! - **Compile-time.** A columnar node's store handle lives behind
//!   `ColumnarRow`, whose `store` field is private to `graph::storage`, and the
//!   `NodeData` property readers Phase 1 emptied are deleted. New code
//!   *cannot* express a direct-route read; it fails to compile. The two named
//!   escapes (`ColumnarRow::node_handle` / `::repoint`) are pinned site-for-site
//!   by `the_node_handle_escape_has_exactly_the_phase_3_call_sites`, so the
//!   remaining direct-route set is an enumerated work list rather than a
//!   guess.
//! - **Runtime**, for what the compiler cannot see: a caller that *could* have
//!   used the accessors but reads a `NodeData` it already holds. `poison_*`
//!   makes the node handle and the backend's store disagree — exactly as
//!   Phase 3 will — and one named test per caller class asserts the class
//!   observes the authoritative value. Each was shown red by reverting that
//!   one call site; see the commit body.
//!
//! # Why the divergence tests do not just assert "the master wins"
//!
//! On HEAD the node handle *is* the read route on memory/mapped
//! (`storage/impls.rs`'s `get_node_property` → `node_weight(idx).properties`),
//! and `refresh_columnar_node_handles` exists precisely to push master writes
//! back onto the nodes at end-of-clause. Asserting master-authority now would
//! be asserting Phase 3's end state, i.e. a permanently red test. The pins
//! below record the current answer so the change is visible when it happens.

use std::collections::HashMap;
use std::sync::Arc;

use crate::datatypes::{DataFrame, Value};
use crate::graph::dir_graph::DirGraph;
use crate::graph::schema::{InternedKey, PropertyStorage};
use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
use crate::graph::storage::column_store::ColumnStore;
use crate::graph::storage::poison::{self, PoisonGuard};
use crate::graph::storage::{GraphRead, GraphWrite};
use petgraph::graph::NodeIndex;

const N: i64 = 4;

fn run(graph: &mut DirGraph, query: &str) {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    execute_mut(graph, query, &opts).unwrap_or_else(|e| panic!("setup query failed: {query}: {e}"));
}

fn read_one(graph: &DirGraph, query: &str) -> Value {
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let out = execute_read(graph, query, &opts).unwrap_or_else(|e| panic!("{query}: {e}"));
    out.result
        .rows
        .first()
        .and_then(|r| r.first())
        .cloned()
        .unwrap_or(Value::Null)
}

/// `N` `Item` nodes with two ordinary properties each, then `enable_columnar()`
/// — the shape every graph takes the moment it is saved.
fn seeded_columnar() -> DirGraph {
    let mut g = DirGraph::new();
    let rows: Vec<Vec<Value>> = (1..=N)
        .map(|i| {
            vec![
                Value::Int64(i),
                Value::String(format!("t{i}")),
                Value::String(format!("c0-{i}")),
                Value::Int64(i * 10),
            ]
        })
        .collect();
    let df = DataFrame::from_cypher_rows(
        vec![
            "id".to_string(),
            "title".to_string(),
            "c0".to_string(),
            "c1".to_string(),
        ],
        rows,
    )
    .unwrap();
    crate::graph::mutation::maintain::add_nodes(
        &mut g,
        df,
        "Item".to_string(),
        "id".to_string(),
        Some("title".to_string()),
        None,
    )
    .unwrap();
    g.enable_columnar();
    assert!(
        !g.column_stores.is_empty(),
        "fixture must own a master column store, or every arm below is vacuous"
    );
    assert!(
        node_row_id(&g, node_of(&g, 1)).is_some(),
        "fixture nodes must hold a columnar handle, or divergence is unconstructible"
    );
    g
}

fn node_of(graph: &DirGraph, id: i64) -> NodeIndex {
    graph
        .graph
        .node_indices()
        .find(|&i| graph.graph.get_node_id(i) == Some(Value::Int64(id)))
        .unwrap_or_else(|| panic!("no Item with id {id}"))
}

fn node_row_id(graph: &DirGraph, idx: NodeIndex) -> Option<u32> {
    match graph.graph.node_weight(idx).map(|n| &n.properties) {
        Some(PropertyStorage::Columnar(row)) => Some(row.row_id()),
        _ => None,
    }
}

/// Does node `idx` still share the master's `Arc`?
fn node_shares_master(graph: &DirGraph, idx: NodeIndex) -> bool {
    let master = graph.column_stores.get("Item").expect("master store");
    match graph.graph.node_weight(idx).map(|n| &n.properties) {
        Some(PropertyStorage::Columnar(row)) => Arc::ptr_eq(row.node_handle(), master),
        _ => false,
    }
}

/// Fork the master away from every node handle and write `value` into it.
///
/// `Arc::make_mut` on the master succeeds in forking precisely because the
/// nodes hold strong handles (D1 §1.2). Returns the interned key written.
fn diverge_master(graph: &mut DirGraph, idx: NodeIndex, key: &str, value: Value) -> InternedKey {
    let row_id = node_row_id(graph, idx).expect("columnar node");
    let ikey = graph.interner.get_or_intern(key);
    let master = Arc::make_mut(graph.column_stores.get_mut("Item").expect("master store"));
    assert!(
        master.set(row_id, ikey, &value, None),
        "master write must land"
    );
    assert!(
        !node_shares_master(graph, idx),
        "the master must have forked away from the node handles, or there is no divergence \
         to observe and every assertion below is vacuous"
    );
    ikey
}

/// Pull `c0` for the node with `id: 1` out of a D3-JSON export.
/// `Value::Null` when the key is absent (which is what a REMOVE produces).
fn extract_json_c0(json: &str) -> Value {
    let obj = json
        .split('{')
        .find(|chunk| chunk.contains("\"id\":1,"))
        .unwrap_or("");
    match obj.split("\"c0\":").nth(1) {
        Some(rest) => {
            let raw = rest
                .split([',', '}'])
                .next()
                .unwrap_or("")
                .trim()
                .trim_matches('"');
            Value::String(raw.to_string())
        }
        None => Value::Null,
    }
}

/// The value each public read surface resolves for `Item{id:1}.c0`.
fn all_read_surfaces(
    graph: &mut DirGraph,
    idx: NodeIndex,
    ikey: InternedKey,
) -> Vec<(&'static str, Value)> {
    // `read_indexed` is the funnel every index / constraint build reads
    // through; its `PropertyReader` needs `&mut` only to intern the key.
    let reader = graph.property_reader("Item", "c0");
    let graph = &*graph;
    vec![
        (
            "GraphRead::node_view",
            graph
                .node_view(idx)
                .and_then(|v| v.get_property("c0"))
                .map(|c| c.into_owned())
                .unwrap_or(Value::Null),
        ),
        (
            "GraphRead::get_node_property",
            graph
                .graph
                .get_node_property(idx, ikey)
                .unwrap_or(Value::Null),
        ),
        (
            "GraphRead::node_row_properties",
            graph
                .graph
                .node_row_properties(idx)
                .into_iter()
                .find(|(k, _)| *k == ikey)
                .map(|(_, v)| v)
                .unwrap_or(Value::Null),
        ),
        (
            "DirGraph::read_indexed (index build funnel)",
            graph.read_indexed(&reader, idx).unwrap_or(Value::Null),
        ),
        (
            "Cypher RETURN n.c0",
            read_one(graph, "MATCH (n:Item) WHERE n.id = 1 RETURN n.c0"),
        ),
        (
            "Cypher RETURN n (whole-node projection)",
            match read_one(graph, "MATCH (n:Item) WHERE n.id = 1 RETURN n") {
                Value::Node(nv) => nv.properties.get("c0").cloned().unwrap_or(Value::Null),
                other => panic!("expected a node value, got {other:?}"),
            },
        ),
        (
            "Cypher properties(n)",
            match read_one(graph, "MATCH (n:Item) WHERE n.id = 1 RETURN properties(n)") {
                Value::Map(m) => m.get("c0").cloned().unwrap_or(Value::Null),
                other => panic!("expected a map, got {other:?}"),
            },
        ),
        ("D3-JSON export", {
            let json = crate::graph::io::export::to_d3_json(graph, None).unwrap();
            extract_json_c0(&json)
        }),
    ]
}

// ── 1. Cross-surface consistency — phase-independent ───────────────────────

/// Without divergence, every surface must see the stored value. Without this
/// arm the consistency test below would pass on a build where every surface
/// returned `Null`.
#[test]
fn all_public_reads_agree_without_divergence() {
    let mut graph = seeded_columnar();
    let idx = node_of(&graph, 1);
    let ikey = InternedKey::from_str("c0");
    let stored = Value::String("c0-1".into());
    for (surface, got) in all_read_surfaces(&mut graph, idx, ikey) {
        assert_eq!(got, stored, "{surface} disagreed with the stored value");
    }
}

/// **The gate.** Under master/node divergence every public read surface must
/// still agree with every other one. Which replica wins is pinned separately;
/// what must never happen is two surfaces answering differently, because that
/// is a user-visible inconsistency no matter which side is authoritative.
#[test]
fn all_public_reads_agree_under_master_node_divergence() {
    let mut graph = seeded_columnar();
    let idx = node_of(&graph, 1);
    let ikey = diverge_master(&mut graph, idx, "c0", Value::String("MASTER".into()));

    let surfaces = all_read_surfaces(&mut graph, idx, ikey);
    let (first_name, first) = surfaces[0].clone();
    for (surface, got) in &surfaces[1..] {
        assert_eq!(
            got, &first,
            "{surface} resolved {got:?} but {first_name} resolved {first:?} — \
             two public reads of the same property must never disagree"
        );
    }
}

// ── 2. Which replica wins — pinned, inverted by Phase 3 ────────────────────

/// Today the node handle wins: memory/mapped `get_node_property` reads
/// `node_weight(idx).properties`, so a master-only write is invisible.
///
/// **Phase 3 inverts this.** When `PropertyStorage::Columnar` loses its `store`
/// field and the backend owns the map, the expected value becomes `"MASTER"`.
/// Getting `"MASTER"` here before then means ownership moved early — invert the
/// assertion, do not delete it.
#[test]
fn today_the_node_handle_wins_over_the_master() {
    let mut graph = seeded_columnar();
    let idx = node_of(&graph, 1);
    let ikey = diverge_master(&mut graph, idx, "c0", Value::String("MASTER".into()));
    assert_eq!(
        graph.graph.get_node_property(idx, ikey),
        Some(Value::String("c0-1".into())),
        "the node handle is the read route on memory/mapped today; Phase 3 makes this MASTER"
    );
}

// ── 3. Writes reconverge the two replicas ──────────────────────────────────

/// A columnar `SET` writes through the master and then re-points every node of
/// the type (`refresh_columnar_node_handles`). After it, master and nodes must
/// share one `Arc` again and every surface must read the new value — including
/// the surfaces that had been reading a diverged replica.
#[test]
fn set_reconverges_master_and_node_handles() {
    let mut graph = seeded_columnar();
    let idx = node_of(&graph, 1);
    let ikey = diverge_master(&mut graph, idx, "c0", Value::String("MASTER".into()));
    assert!(!node_shares_master(&graph, idx));

    run(
        &mut graph,
        "MATCH (n:Item) WHERE n.id = 1 SET n.c0 = 'WRITTEN'",
    );

    assert!(
        node_shares_master(&graph, idx),
        "a columnar SET must leave the node pointing at the master again"
    );
    let want = Value::String("WRITTEN".into());
    for (surface, got) in all_read_surfaces(&mut graph, idx, ikey) {
        assert_eq!(got, want, "{surface} did not observe the SET");
    }
}

/// The same for `REMOVE`, which takes a different master path
/// (`Arc::make_mut(master).set(.., Null, ..)`).
#[test]
fn remove_reconverges_master_and_node_handles() {
    let mut graph = seeded_columnar();
    let idx = node_of(&graph, 1);
    let ikey = diverge_master(&mut graph, idx, "c0", Value::String("MASTER".into()));

    run(&mut graph, "MATCH (n:Item) WHERE n.id = 1 REMOVE n.c0");

    assert!(
        node_shares_master(&graph, idx),
        "a columnar REMOVE must leave the node pointing at the master again"
    );
    for (surface, got) in all_read_surfaces(&mut graph, idx, ikey) {
        assert_eq!(got, Value::Null, "{surface} still sees a removed property");
    }
}

/// A `MERGE` key read must resolve the same value the read surfaces do —
/// otherwise `MERGE` would create a duplicate for a row that already matches
/// (or match a row that does not).
#[test]
fn merge_key_read_matches_the_public_read() {
    let mut graph = seeded_columnar();
    let idx = node_of(&graph, 1);
    let _ = diverge_master(&mut graph, idx, "c0", Value::String("MASTER".into()));

    let before = graph.graph.node_count();
    let observed = read_one(&graph, "MATCH (n:Item) WHERE n.id = 1 RETURN n.c0");
    let observed_str = match &observed {
        Value::String(s) => s.clone(),
        other => panic!("expected a string, got {other:?}"),
    };
    run(
        &mut graph,
        &format!("MERGE (n:Item {{id: 1, c0: '{observed_str}'}})"),
    );
    assert_eq!(
        graph.graph.node_count(),
        before,
        "MERGE on the value the public read reports must match the existing row, not create one"
    );
}

/// A rolled-back statement must leave every surface on the pre-statement value.
/// The columnar SET path emits no `NodeWeight` undo entry — its only signal is
/// `UndoEntry::ColumnarHandles` — so this is the arm that proves the journal
/// covers the master write at all.
#[test]
fn rollback_restores_every_read_surface() {
    let mut graph = seeded_columnar();
    let idx = node_of(&graph, 1);
    let ikey = InternedKey::from_str("c0");
    let before = graph.graph.get_node_property(idx, ikey);

    // Two patterns: the first commits its SET, the second is rejected, so the
    // whole statement rolls back.
    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let err = execute_mut(
        &mut graph,
        "MATCH (n:Item) WHERE n.id = 1 SET n.c0 = 'DOOMED', \
         n.c1 = duration({months: 2147483648})",
        &opts,
    );
    assert!(err.is_err(), "the fixture statement must fail to roll back");

    assert_eq!(
        graph.graph.get_node_property(idx, ikey),
        before,
        "a rolled-back columnar SET must restore the pre-statement value"
    );
    for (surface, got) in all_read_surfaces(&mut graph, idx, ikey) {
        assert_eq!(
            Some(got),
            before.clone(),
            "{surface} kept a rolled-back value"
        );
    }
}

/// Save + reload must round-trip whatever the read surfaces report — a
/// divergence that only the writer can see is a data-loss bug, not a caching
/// one.
#[test]
fn save_and_reload_round_trips_the_observed_value() {
    let mut graph = seeded_columnar();
    let idx = node_of(&graph, 1);
    let _ = diverge_master(&mut graph, idx, "c0", Value::String("MASTER".into()));
    run(
        &mut graph,
        "MATCH (n:Item) WHERE n.id = 1 SET n.c0 = 'PERSISTED'",
    );

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g.kgl");
    let mut arc = Arc::new(graph);
    crate::graph::io::file::prepare_save(&mut arc);
    Arc::make_mut(&mut arc).enable_columnar();
    crate::graph::io::file::write_kgl(&arc, path.to_str().unwrap()).unwrap();

    let loaded = crate::graph::io::file::load_file(path.to_str().unwrap()).unwrap();
    assert_eq!(
        read_one(&loaded, "MATCH (n:Item) WHERE n.id = 1 RETURN n.c0"),
        Value::String("PERSISTED".into()),
        "the saved file must carry the value the reads reported"
    );
}

// ── 4. Defect 2 — `maybe_spill_columns` reclaims nothing ───────────────────

/// **D1 defect 2, pinned in its current (wrong) state.**
///
/// `maybe_spill_columns` (`dir_graph/mod.rs`) calls
/// `Arc::make_mut(self.column_stores.get_mut(..))` and then
/// `materialize_to_files`. Every node of the type holds a strong handle, so
/// `make_mut` **forks**: the master becomes the file-backed copy while all N
/// nodes keep the pre-spill in-heap store alive, and — unlike the SET path —
/// there is no refresh sweep here to re-point them. Reads stay correct; the
/// memory the spill exists to reclaim is not reclaimed.
///
/// A true red for this needs Phase 3's end state, because the *only* way to
/// make the fork not happen is for the nodes to stop holding handles — which is
/// exactly what Phase 3 does and what Phase 1 is forbidden from doing. So this
/// asserts today's wrong behaviour, exactly:
///
/// - the master is mapped (spilled) — the spill itself ran;
/// - the node still points at a **different**, **unmapped** store — nothing was
///   reclaimed.
///
/// **Phase 3 inverts this**: assert `node_shares_master` and that the node's
/// store is mapped, i.e. `is_mapped()` on both. Getting there early means the
/// defect was fixed — invert the assertion, do not delete it.
#[test]
fn spill_forks_the_master_and_reclaims_nothing_today() {
    let mut graph = seeded_columnar();
    let idx = node_of(&graph, 1);
    let dir = tempfile::tempdir().unwrap();
    graph.spill_dir = Some(dir.path().to_path_buf());
    // Any limit below the store's heap footprint forces a spill.
    graph.memory_limit = Some(0);

    assert!(
        node_shares_master(&graph, idx),
        "precondition: nodes share the master before the spill"
    );

    graph.maybe_spill_columns();

    let master = graph.column_stores.get("Item").expect("master store");
    assert!(
        master.is_mapped(),
        "the spill must have materialised the master to files, or this test proves nothing"
    );
    assert!(
        !node_shares_master(&graph, idx),
        "TODAY: make_mut forks the master away from the node handles. If this fails, \
         the spill now re-points the nodes — invert this test (assert the node shares \
         the mapped master) and close D1 defect 2."
    );
    let node_store_is_mapped = match graph.graph.node_weight(idx).map(|n| &n.properties) {
        Some(PropertyStorage::Columnar(row)) => row.node_handle().is_mapped(),
        _ => panic!("node must still be columnar"),
    };
    assert!(
        !node_store_is_mapped,
        "TODAY: every node keeps the pre-spill in-heap store alive, so the spill reclaims \
         no memory. If this fails, the reclaim works — invert and close D1 defect 2."
    );

    // The user-visible contract is unaffected either way: reads still resolve.
    assert_eq!(
        read_one(&graph, "MATCH (n:Item) WHERE n.id = 1 RETURN n.c0"),
        Value::String("c0-1".into()),
        "a spill must never change what a read returns"
    );
}

// ── 5. Defect 1 — columnar enumeration completeness ────────────────────────

/// `describe()`'s per-type property block and node samples read through the
/// accessors now; before D1 Phase 1 they went through `NodeData::property_iter`
/// and enumerated **nothing** for a saved graph.
#[test]
fn describe_reports_columnar_properties() {
    use crate::graph::introspection::{ConnectionDetail, CypherDetail, FluentDetail};
    let graph = seeded_columnar();
    let xml = crate::graph::introspection::describe::compute_description(
        &graph,
        None,
        &ConnectionDetail::Off,
        &CypherDetail::Off,
        &FluentDetail::Off,
        None,
        None,
        None,
    )
    .unwrap();
    assert!(
        xml.contains("c0"),
        "describe() lost a columnar property: {xml}"
    );
    assert!(
        xml.contains("c0-1"),
        "describe()'s node sample lost a columnar property value: {xml}"
    );
}

/// `compute_property_stats` accumulates from the row, not just from the
/// `type_schemas` pre-seed: a columnar property must report a non-zero
/// non-null count and real sample values.
#[test]
fn property_stats_count_columnar_rows() {
    let graph = seeded_columnar();
    let stats = crate::graph::introspection::schema_overview::compute_property_stats(
        &graph, "Item", 32, None,
    )
    .expect("property stats");
    let c0 = stats
        .iter()
        .find(|p| p.property_name == "c0")
        .expect("c0 must appear in the property stats");
    assert_eq!(
        c0.non_null, N as usize,
        "columnar rows contributed no values to the property stats"
    );
    assert_eq!(
        c0.unique, N as usize,
        "columnar rows contributed no distinct values"
    );
}

/// `property_ndv` — the planner's selectivity input — must see columnar rows.
/// It bypasses `read_indexed` and reads the node directly, so it is one of the
/// callers the inventory flagged as "a reader would assume it is covered".
#[test]
fn property_ndv_counts_columnar_rows() {
    let graph = seeded_columnar();
    assert_eq!(
        graph.property_ndv("Item", "c0"),
        Some(N as usize),
        "property_ndv must see a columnar type's distinct values"
    );
}

// ══════════════════════════════════════════════════════════════════════════
// Phase 2 — the mutation-proof gate
// ══════════════════════════════════════════════════════════════════════════

// ── The poison primitive ──────────────────────────────────────────────────

/// Make the node-held handle and the backend's store disagree, the way D1
/// Phase 3 will, and install the backend's store as the authoritative route.
///
/// Three steps, mirroring `storage/poison.rs`:
/// 1. every node of `node_type` is re-pointed at a **stale** private clone;
/// 2. `edit` writes the truth into the type's **master** store only;
/// 3. the post-write master becomes the authoritative read route.
///
/// A caller that reads through `NodeView` / `GraphRead` sees the truth. One
/// that reads a `NodeData` it already holds sees the stale replica. The guard
/// restores normal resolution on drop — bind it, never `let _ = …`.
fn poison_row(
    graph: &mut DirGraph,
    node_type: &str,
    edit: impl FnOnce(&mut ColumnStore),
) -> PoisonGuard {
    let master_arc = Arc::clone(
        graph
            .column_stores
            .get(node_type)
            .expect("type must be columnar, or the poison is a no-op"),
    );
    let stale = Arc::new((*master_arc).clone());

    let indices: Vec<NodeIndex> = graph
        .type_indices
        .get(node_type)
        .map(|set| set.iter().collect())
        .unwrap_or_default();
    assert!(
        !indices.is_empty(),
        "no nodes of type {node_type} to poison"
    );
    for idx in indices {
        if let Some(node) = GraphWrite::node_weight_mut_silent(&mut graph.graph, idx) {
            if let PropertyStorage::Columnar(row) = &mut node.properties {
                row.repoint(Arc::clone(&stale));
            }
        }
    }

    let master = Arc::make_mut(graph.column_stores.get_mut(node_type).unwrap());
    edit(master);

    // Leaked because `NodeView` hands out `&ColumnStore` borrows; one snapshot
    // per poison call is bounded by the number of tests here.
    let leaked: &'static ColumnStore = Box::leak(Box::new(master.clone()));
    poison::install(InternedKey::from_str(node_type), leaked)
}

/// Poison one row's property column.
fn poison_property(
    graph: &mut DirGraph,
    node_type: &str,
    row_id: u32,
    key: &str,
    value: Value,
) -> PoisonGuard {
    let ikey = graph.interner.get_or_intern(key);
    poison_row(graph, node_type, move |store| {
        assert!(
            store.set(row_id, ikey, &value, None),
            "master write must land, or the poison proves nothing"
        );
    })
}

/// Poison one row's `__title__` column.
fn poison_title(graph: &mut DirGraph, node_type: &str, row_id: u32, value: Value) -> PoisonGuard {
    poison_row(graph, node_type, move |store| {
        assert!(
            store.set_title(row_id, &value),
            "master title write must land, or the poison proves nothing"
        );
    })
}

/// The value the node's **own** handle still reports — the stale replica.
fn stale_node_route(graph: &DirGraph, idx: NodeIndex, key: &str) -> Option<Value> {
    match graph.graph.node_weight(idx).map(|n| &n.properties) {
        Some(PropertyStorage::Columnar(row)) => row
            .node_handle()
            .get(row.row_id(), InternedKey::from_str(key)),
        _ => None,
    }
}

/// Fixture: a saved graph with row 0 (`id: 1`) poisoned so its authoritative
/// `c0` is `TRUTH` while its node handle still says `c0-1`.
fn poisoned_fixture() -> (DirGraph, NodeIndex, PoisonGuard) {
    let mut graph = seeded_columnar();
    let idx = node_of(&graph, 1);
    let row_id = node_row_id(&graph, idx).expect("columnar node");
    let guard = poison_property(
        &mut graph,
        "Item",
        row_id,
        "c0",
        Value::String("TRUTH".into()),
    );
    (graph, idx, guard)
}

/// **Non-vacuity.** The poison must actually split the two routes; if it did
/// not, every class test below would pass by accident.
#[test]
fn poison_makes_the_node_route_and_the_authoritative_route_disagree() {
    let (graph, idx, _guard) = poisoned_fixture();
    assert_eq!(
        graph.node_view(idx).unwrap().get_property_value("c0"),
        Some(Value::String("TRUTH".into())),
        "the accessor route must see the authoritative value"
    );
    assert_eq!(
        stale_node_route(&graph, idx, "c0"),
        Some(Value::String("c0-1".into())),
        "the node's own handle must still hold the stale replica — without this \
         the poison is a no-op and every class test below is vacuous"
    );
}

/// The guard must restore normal resolution, or poison would leak across tests.
#[test]
fn dropping_the_poison_guard_restores_node_handle_resolution() {
    let mut graph = seeded_columnar();
    let idx = node_of(&graph, 1);
    let row_id = node_row_id(&graph, idx).unwrap();
    {
        let _guard = poison_property(
            &mut graph,
            "Item",
            row_id,
            "c0",
            Value::String("TRUTH".into()),
        );
        assert_eq!(
            graph.node_view(idx).unwrap().get_property_value("c0"),
            Some(Value::String("TRUTH".into()))
        );
    }
    assert_eq!(
        graph.node_view(idx).unwrap().get_property_value("c0"),
        Some(Value::String("c0-1".into())),
        "resolution must fall back to the node handle once the guard is dropped"
    );
}

// ── One named test per caller class ───────────────────────────────────────

/// **R1 — pattern matcher filter.** `MATCH (n:Item {c0: …})` resolves the
/// authoritative value, so the inline-property filter finds the poisoned row
/// and not the stale one.
#[test]
fn r1_matcher_property_filter_reads_the_authoritative_store() {
    let (graph, _idx, _guard) = poisoned_fixture();
    assert_eq!(
        read_one(&graph, "MATCH (n:Item {c0: 'TRUTH'}) RETURN n.id"),
        Value::Int64(1),
        "the matcher's property filter must see the authoritative value"
    );
    assert_eq!(
        read_one(&graph, "MATCH (n:Item {c0: 'c0-1'}) RETURN n.id"),
        Value::Null,
        "the matcher must not match the stale replica"
    );
}

/// **R3 — WHERE / expression resolution.**
#[test]
fn r3_where_clause_reads_the_authoritative_store() {
    let (graph, _idx, _guard) = poisoned_fixture();
    assert_eq!(
        read_one(&graph, "MATCH (n:Item) WHERE n.c0 = 'TRUTH' RETURN n.id"),
        Value::Int64(1)
    );
    assert_eq!(
        read_one(&graph, "MATCH (n:Item) WHERE n.id = 1 RETURN n.c0"),
        Value::String("TRUTH".into())
    );
}

/// **R4 — projection / whole-node materialisation.**
#[test]
fn r4_whole_node_projection_reads_the_authoritative_store() {
    let (graph, _idx, _guard) = poisoned_fixture();
    match read_one(&graph, "MATCH (n:Item) WHERE n.id = 1 RETURN n") {
        Value::Node(nv) => assert_eq!(
            nv.properties.get("c0"),
            Some(&Value::String("TRUTH".into())),
            "RETURN n must carry the authoritative value"
        ),
        other => panic!("expected a node value, got {other:?}"),
    }
}

/// **R8 — index build funnel (`read_indexed`).** A property index built after
/// the poison buckets the row under its authoritative value.
#[test]
fn r8_property_index_build_reads_the_authoritative_store() {
    let (mut graph, idx, _guard) = poisoned_fixture();
    graph.create_index("Item", "c0");
    let bucket = graph
        .property_indices
        .get(&("Item".to_string(), "c0".to_string()))
        .expect("index must exist");
    assert_eq!(
        bucket.get(&Value::String("TRUTH".into())),
        Some(&vec![idx]),
        "the built index must bucket the row under its authoritative value"
    );
    assert!(
        !bucket.contains_key(&Value::String("c0-1".into())),
        "the built index must not carry the stale replica's value"
    );
}

/// **R9 — incremental index maintenance.** The incremental updater
/// (`update_property_indices_for_add`) bypasses `read_indexed`, so it gets its
/// own arm: it must agree with the rebuild above.
#[test]
fn r9_incremental_index_maintenance_reads_the_authoritative_store() {
    let mut graph = seeded_columnar();
    let idx = node_of(&graph, 1);
    let row_id = node_row_id(&graph, idx).unwrap();
    // Build the index *before* the poison, from the pre-poison values.
    graph.create_index("Item", "c0");
    let _guard = poison_property(
        &mut graph,
        "Item",
        row_id,
        "c0",
        Value::String("TRUTH".into()),
    );

    graph.update_property_indices_for_add("Item", idx);

    let bucket = graph
        .property_indices
        .get(&("Item".to_string(), "c0".to_string()))
        .expect("index must exist");
    assert!(
        bucket
            .get(&Value::String("TRUTH".into()))
            .is_some_and(|members| members.contains(&idx)),
        "incremental maintenance must file the row under its authoritative \
         value, or it disagrees with a rebuilt index"
    );
}

/// **R11 — constraint gates.** Declaring a unique constraint validates the
/// existing rows through `read_indexed`; with two rows sharing an
/// authoritative `c0`, the declaration must be rejected.
#[test]
fn r11_unique_constraint_gate_reads_the_authoritative_store() {
    let mut graph = seeded_columnar();
    let idx = node_of(&graph, 1);
    let row_id = node_row_id(&graph, idx).unwrap();
    // Collide row 0 with row 1's value (`c0-2`) in the master only.
    let _guard = poison_property(
        &mut graph,
        "Item",
        row_id,
        "c0",
        Value::String("c0-2".into()),
    );

    let params = HashMap::new();
    let opts = ExecuteOptions::eager(&params);
    let result = execute_mut(
        &mut graph,
        "CREATE CONSTRAINT FOR (i:Item) REQUIRE i.c0 IS UNIQUE",
        &opts,
    );
    assert!(
        result.is_err(),
        "the constraint gate must see the authoritative duplicate and reject; \
         reading the stale node handles would show four distinct values"
    );
}

/// **R12 — planner statistics.** `property_ndv` bypasses `read_indexed`, so it
/// gets its own arm: the collision above must drop the distinct count.
#[test]
fn r12_property_ndv_reads_the_authoritative_store() {
    let mut graph = seeded_columnar();
    let idx = node_of(&graph, 1);
    let row_id = node_row_id(&graph, idx).unwrap();
    let _guard = poison_property(
        &mut graph,
        "Item",
        row_id,
        "c0",
        Value::String("c0-2".into()),
    );
    assert_eq!(
        graph.property_ndv("Item", "c0"),
        Some(N as usize - 1),
        "property_ndv must count the authoritative values; reading the stale \
         node handles would still report {N} distinct"
    );
}

/// **R13a — export.** The D3-JSON exporter enumerates a node's properties.
#[test]
fn r13_export_reads_the_authoritative_store() {
    let (graph, _idx, _guard) = poisoned_fixture();
    let json = crate::graph::io::export::to_d3_json(&graph, None).unwrap();
    assert_eq!(
        extract_json_c0(&json),
        Value::String("TRUTH".into()),
        "D3-JSON export must carry the authoritative value"
    );
}

/// **R13b — introspection statistics.** Asserted on `compute_property_stats`
/// directly rather than on the rendered `describe()` XML: the XML mentions a
/// value in several places, so a substring check there cannot tell which
/// producer supplied it, and a mutation of the stats accumulator left it green.
#[test]
fn r13_property_stats_read_the_authoritative_store() {
    let (graph, _idx, _guard) = poisoned_fixture();
    let stats = crate::graph::introspection::schema_overview::compute_property_stats(
        &graph, "Item", 32, None,
    )
    .expect("property stats");
    let c0 = stats
        .iter()
        .find(|p| p.property_name == "c0")
        .expect("c0 must appear in the property stats");
    let values = c0.values.as_ref().expect("small-cardinality values");
    assert!(
        values.contains(&Value::String("TRUTH".into())),
        "property stats must observe the authoritative value; got {values:?}"
    );
    assert!(
        !values.contains(&Value::String("c0-1".into())),
        "property stats must not observe the stale replica; got {values:?}"
    );
}

/// **R14 — binding-layer readers.** `session::resolve_noderefs` is public API,
/// runs after the executor returns and holds only a `&GraphBackend`.
#[test]
fn r14_resolve_noderefs_reads_the_authoritative_store() {
    let mut graph = seeded_columnar();
    let idx = node_of(&graph, 1);
    let row_id = node_row_id(&graph, idx).unwrap();
    let _guard = poison_title(
        &mut graph,
        "Item",
        row_id,
        Value::String("TRUE-TITLE".into()),
    );

    let mut rows = vec![vec![Value::NodeRef(idx.index() as u32)]];
    crate::graph::session::resolve_noderefs(&graph.graph, &mut rows);
    assert_eq!(
        rows[0][0],
        Value::String("TRUE-TITLE".into()),
        "resolve_noderefs must resolve the authoritative title"
    );
}

// ── The compile-time gate's enumerated escape list ────────────────────────

/// The **only** two ways to reach a node's own store handle outside
/// `graph::storage`, and the exact set of call sites, file by file.
///
/// This *is* the D1 Phase-3 work list for the node-held handle: when the
/// `store` field is deleted, these are the sites that must change, and there
/// are no others because nothing else can express the read.
const NODE_HANDLE_ESCAPE_SITES: &[(&str, usize)] = &[
    // W8 — `enable_columnar` drift check + rebuild, `disable_columnar` restore.
    ("graph/dir_graph/mod.rs", 5),
    // W10 — the rollback restore arm re-points nodes at the pre-statement store.
    ("graph/dir_graph/rollback.rs", 1),
    // W5 — the end-of-clause refresh sweep.
    ("graph/languages/cypher/executor/columnar_write.rs", 1),
    // Tests that are *about* handle identity, and must be rewritten (not
    // deleted) when the handle goes — see the plan's Phase 3 step 5.
    ("graph/dir_graph/rollback_tests.rs", 2),
    ("graph/storage/column_ownership_tests.rs", 4),
];

/// Pins [`NODE_HANDLE_ESCAPE_SITES`] against the source tree.
///
/// Fails on a **new** call site (someone re-opened the direct route) and on a
/// **removed** one (Phase 3 landed, or coverage was lost) — so it cannot rot in
/// either direction. Update the table in the same change that moves a site.
#[test]
fn the_node_handle_escape_has_exactly_the_phase_3_call_sites() {
    use std::collections::BTreeMap;

    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut found: BTreeMap<String, usize> = BTreeMap::new();

    // Split so this detector does not match its own source line — otherwise
    // the file's count would include the scanner and drift every time this
    // function is edited.
    let read_escape = concat!(".node_", "handle()");
    let write_escape = concat!(".re", "point(");

    fn walk(
        dir: &std::path::Path,
        root: &std::path::Path,
        needles: (&str, &str),
        found: &mut BTreeMap<String, usize>,
    ) {
        for entry in std::fs::read_dir(dir).expect("readable source dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                walk(&path, root, needles, found);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let text = std::fs::read_to_string(&path).expect("readable source file");
                let hits = text.matches(needles.0).count() + text.matches(needles.1).count();
                if hits > 0 {
                    let rel = path
                        .strip_prefix(root)
                        .expect("under src")
                        .to_string_lossy()
                        .replace('\\', "/");
                    *found.entry(rel).or_insert(0) += hits;
                }
            }
        }
    }
    walk(&src, &src, (read_escape, write_escape), &mut found);

    let expected: BTreeMap<String, usize> = NODE_HANDLE_ESCAPE_SITES
        .iter()
        .map(|(f, n)| ((*f).to_string(), *n))
        .collect();

    assert_eq!(
        found, expected,
        "\nThe node-held column-store handle is reachable from a set of call \
         sites that no longer matches the D1 Phase-3 work list.\n\
         - A NEW entry means the direct route was re-opened: read through \
           `NodeView` / `GraphRead` instead.\n\
         - A MISSING or smaller entry means a Phase-3 site was removed: update \
           `NODE_HANDLE_ESCAPE_SITES` in the same change.\n"
    );
}
