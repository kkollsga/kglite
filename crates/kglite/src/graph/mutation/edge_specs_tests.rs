use super::*;
use crate::datatypes::DataFrame;
use crate::graph::mutation::maintain::{add_connections, add_nodes};
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};
use crate::graph::storage::GraphRead;
use tempfile::TempDir;

fn add_pair(graph: &mut DirGraph) {
    let rows = DataFrame::from_cypher_rows(
        vec!["id".to_string()],
        vec![vec![Value::Int64(1)], vec![Value::Int64(2)]],
    )
    .unwrap();
    add_nodes(
        graph,
        rows,
        "Doc".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .unwrap();
}

fn link(source_id: i64, target_id: i64) -> EdgeSpec {
    EdgeSpec {
        source_type: "Doc".to_string(),
        source_id: Value::Int64(source_id),
        target_type: "Doc".to_string(),
        target_id: Value::Int64(target_id),
        edge_type: "LINKS".to_string(),
        properties: HashMap::from([("weight".to_string(), Value::Int64(3))]),
    }
}

#[test]
fn edge_specs_mutate_every_storage_mode_and_invalidate_counts() {
    for mode in [StorageMode::Memory, StorageMode::Mapped, StorageMode::Disk] {
        let tmp = TempDir::new().unwrap();
        let path = (mode == StorageMode::Disk).then_some(tmp.path());
        let mut graph = new_dir_graph_in_mode(mode, path).unwrap();
        add_pair(&mut graph);
        *graph.edge_type_counts_cache.write().unwrap() = Some(std::sync::Arc::new(HashMap::from(
            [("LINKS".to_string(), 0usize)],
        )));

        let report = add_edges_from_specs(&mut graph, vec![link(1, 2), link(99, 2)]).unwrap();

        assert_eq!(report.connections_created, 1, "mode={mode:?}");
        assert_eq!(report.skipped_missing_endpoint, 1, "mode={mode:?}");
        assert_eq!(graph.graph.edge_count(), 1, "mode={mode:?}");
        assert!(!graph.has_edge_type_counts_cache(), "mode={mode:?}");
    }
}

#[test]
fn edge_specs_interner_preflight_is_atomic() {
    let mut graph = DirGraph::new();
    add_pair(&mut graph);
    let incoming = "LINKS";
    graph
        .interner
        .try_register(InternedKey::from_str(incoming), "conflicting-existing")
        .unwrap();

    let error = add_edges_from_specs(&mut graph, vec![link(1, 2)]).unwrap_err();

    assert!(error.contains("hash collision"));
    assert_eq!(graph.graph.edge_count(), 0);
    assert!(graph.connection_type_metadata.is_empty());
}

// ── declared relationship constraints ────────────────────────────────
//
// The spec path must judge a batch exactly as `add_connections` does: the
// C ABI (`kglite_create_edges_batch`) and `from_records` with a `drop`/`error`
// endpoint policy reach the graph only through here.

fn spec(source_id: i64, target_id: i64, edge_type: &str, props: &[(&str, Value)]) -> EdgeSpec {
    EdgeSpec {
        source_type: "Doc".to_string(),
        source_id: Value::Int64(source_id),
        target_type: "Doc".to_string(),
        target_id: Value::Int64(target_id),
        edge_type: edge_type.to_string(),
        properties: props
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    }
}

fn three_docs() -> DirGraph {
    let mut graph = DirGraph::new();
    let rows = DataFrame::from_cypher_rows(
        vec!["id".to_string()],
        (1..=3).map(|id| vec![Value::Int64(id)]).collect(),
    )
    .unwrap();
    add_nodes(
        &mut graph,
        rows,
        "Doc".to_string(),
        "id".to_string(),
        Some("id".to_string()),
        None,
    )
    .unwrap();
    graph
}

fn require_since(graph: &mut DirGraph) {
    graph
        .create_rel_not_null_constraint(
            "LINKS",
            "since",
            &crate::graph::algorithms::Interrupt::default(),
        )
        .unwrap();
}

fn integer_weight(graph: &mut DirGraph) {
    graph
        .create_rel_property_type_constraint(
            "LINKS",
            "weight",
            crate::graph::property_types::DeclaredType::Integer,
            &crate::graph::algorithms::Interrupt::default(),
        )
        .unwrap();
}

#[test]
fn edge_specs_refuse_a_missing_required_relationship_property() {
    let mut graph = three_docs();
    require_since(&mut graph);

    let error = add_edges_from_specs(
        &mut graph,
        vec![
            spec(1, 2, "LINKS", &[("since", Value::Int64(2020))]),
            spec(2, 3, "LINKS", &[("weight", Value::Int64(1))]),
        ],
    )
    .expect_err("a LINKS without `since` violates the NOT NULL constraint");

    assert!(error.contains("LINKS.since"), "{error}");
    assert_eq!(graph.graph.edge_count(), 0, "the refusal is all-or-nothing");
}

#[test]
fn edge_specs_refuse_a_wrongly_typed_relationship_property() {
    let mut graph = three_docs();
    integer_weight(&mut graph);

    let error = add_edges_from_specs(
        &mut graph,
        vec![spec(
            1,
            2,
            "LINKS",
            &[("weight", Value::String("heavy".into()))],
        )],
    )
    .expect_err("a STRING weight violates the IS :: INTEGER constraint");

    assert!(error.contains("STRING"), "{error}");
    assert!(error.contains("LINKS.weight"), "{error}");
    assert_eq!(graph.graph.edge_count(), 0);
}

/// A violation in a later group refuses the whole call — an earlier group of a
/// different, unconstrained edge type must not have been written.
#[test]
fn edge_specs_refusal_in_a_later_group_writes_no_earlier_group() {
    let mut graph = three_docs();
    require_since(&mut graph);

    // Groups run in `(source, target, edge type)` order: CITES before LINKS.
    let error = add_edges_from_specs(
        &mut graph,
        vec![spec(1, 2, "CITES", &[]), spec(2, 3, "LINKS", &[])],
    )
    .expect_err("the LINKS row violates NOT NULL");

    assert!(error.contains("LINKS.since"), "{error}");
    assert_eq!(graph.graph.edge_count(), 0);
    assert!(!graph.connection_type_metadata.contains_key("CITES"));
}

/// The spec path merges under `update`: a row that omits the required
/// property merges into a stored edge that has it, so the result is legal.
#[test]
fn edge_specs_admit_an_update_that_keeps_the_stored_required_value() {
    let mut graph = three_docs();
    add_edges_from_specs(
        &mut graph,
        vec![spec(1, 2, "LINKS", &[("since", Value::Int64(2020))])],
    )
    .unwrap();
    require_since(&mut graph);

    let report = add_edges_from_specs(
        &mut graph,
        vec![spec(1, 2, "LINKS", &[("weight", Value::Int64(5))])],
    )
    .expect("the stored `since` survives an update merge");

    assert_eq!(report.connections_created, 0);
    assert_eq!(report.connections_updated, 1);
    assert_eq!(graph.graph.edge_count(), 1);
}

/// A legal batch still lands under both constraints, with the counters
/// `add_connections` reports for the same rows.
#[test]
fn edge_specs_legal_batch_matches_add_connections_counts() {
    let rows = [(1, 2, 2020, 1), (2, 3, 2021, 2), (1, 3, 2022, 3)];

    let mut via_specs = three_docs();
    require_since(&mut via_specs);
    integer_weight(&mut via_specs);
    let report = add_edges_from_specs(
        &mut via_specs,
        rows.iter()
            .map(|(s, t, since, weight)| {
                spec(
                    *s,
                    *t,
                    "LINKS",
                    &[
                        ("since", Value::Int64(*since)),
                        ("weight", Value::Int64(*weight)),
                    ],
                )
            })
            .collect(),
    )
    .unwrap();

    let mut via_frame = three_docs();
    require_since(&mut via_frame);
    integer_weight(&mut via_frame);
    let frame = DataFrame::from_cypher_rows(
        ["s", "t", "since", "weight"].map(String::from).to_vec(),
        rows.iter()
            .map(|(s, t, since, weight)| {
                vec![
                    Value::Int64(*s),
                    Value::Int64(*t),
                    Value::Int64(*since),
                    Value::Int64(*weight),
                ]
            })
            .collect(),
    )
    .unwrap();
    let frame_report = add_connections(
        &mut via_frame,
        frame,
        "LINKS".to_string(),
        "Doc".to_string(),
        "s".to_string(),
        "Doc".to_string(),
        "t".to_string(),
        None,
        None,
        None,
    )
    .unwrap();

    assert_eq!(report.connections_created, 3);
    assert_eq!(report.connections_created, frame_report.connections_created);
    assert_eq!(report.connections_updated, frame_report.connections_updated);
    assert_eq!(report.skipped_missing_endpoint, 0);
    assert_eq!(via_specs.graph.edge_count(), via_frame.graph.edge_count());
}

// ── shared relationship types ────────────────────────────────────────
//
// Ownership is per (edge type, source node type), as for `add_connections`:
// a group whose source type the edge type has no edge from yet writes one
// edge per spec; a group from a source type it has seen merges.

fn licensee(source_type: &str, source_id: i64, from: i64) -> EdgeSpec {
    EdgeSpec {
        source_type: source_type.to_string(),
        source_id: Value::Int64(source_id),
        target_type: "Company".to_string(),
        target_id: Value::Int64(1),
        edge_type: "HAS_LICENSEE".to_string(),
        properties: HashMap::from([("vf".to_string(), Value::Int64(from))]),
    }
}

fn add_typed(graph: &mut DirGraph, node_type: &str, id: i64) {
    let rows =
        DataFrame::from_cypher_rows(vec!["id".to_string()], vec![vec![Value::Int64(id)]]).unwrap();
    add_nodes(
        graph,
        rows,
        node_type.to_string(),
        "id".to_string(),
        None,
        None,
    )
    .unwrap();
}

#[test]
fn edge_specs_a_second_source_type_owns_its_rows_in_every_storage_mode() {
    for mode in [StorageMode::Memory, StorageMode::Mapped, StorageMode::Disk] {
        let tmp = TempDir::new().unwrap();
        let path = (mode == StorageMode::Disk).then_some(tmp.path());
        let mut graph = new_dir_graph_in_mode(mode, path).unwrap();
        add_typed(&mut graph, "Company", 1);
        add_typed(&mut graph, "Field", 10);
        add_typed(&mut graph, "Licence", 50);

        // The type's first appearance, from two source types in one call:
        // each group owns its repeated pair.
        let first = add_edges_from_specs(
            &mut graph,
            vec![
                licensee("Field", 10, 2001),
                licensee("Field", 10, 2005),
                licensee("Licence", 50, 2001),
                licensee("Licence", 50, 2004),
            ],
        )
        .unwrap();
        assert_eq!(
            (first.connections_created, first.connections_updated),
            (4, 0),
            "mode={mode:?}"
        );

        // A later call from a source type the edge type has not seen owns its
        // rows too, though the type is registered.
        add_typed(&mut graph, "Block", 70);
        let block = add_edges_from_specs(
            &mut graph,
            vec![licensee("Block", 70, 2001), licensee("Block", 70, 2002)],
        )
        .unwrap();
        assert_eq!(
            (block.connections_created, block.connections_updated),
            (2, 0),
            "mode={mode:?}"
        );

        // A re-load from a source type it has seen merges.
        let reload = add_edges_from_specs(&mut graph, vec![licensee("Licence", 50, 2007)]).unwrap();
        assert_eq!(
            (reload.connections_created, reload.connections_updated),
            (0, 1),
            "mode={mode:?}"
        );
        assert_eq!(graph.graph.edge_count(), 6, "mode={mode:?}");
    }
}
