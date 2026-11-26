//! The subgraph copy owns exactly its kept rows, and reads back what the
//! per-node copy reads back, in every storage mode.

use std::collections::HashMap;

use super::*;
use crate::graph::schema::{NodeSchemaDefinition, SchemaDefinition};
use crate::graph::session::execute::{execute_mut, ExecuteOptions};
use crate::graph::storage::column_store::TypedColumn;
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};

/// A relationship as (source, target, type, properties).
type EdgeRecord = (usize, usize, String, Vec<(InternedKey, Value)>);

const MODES: [StorageMode; 3] = [StorageMode::Memory, StorageMode::Mapped, StorageMode::Disk];

fn run(g: &mut DirGraph, query: &str, params: &HashMap<String, Value>) {
    execute_mut(g, query, &ExecuteOptions::eager(params)).expect(query);
}

/// `n` `T` nodes covering every column kind — int, float, string, date, bool,
/// a list (`Mixed`), a column set on one row in a thousand — with string
/// rewrites in the relocation overlay, a float column widened to `Mixed` by an
/// integer it cannot hold exactly, a second type `U`, and `R`/`S`
/// relationships.
fn graph(mode: StorageMode, dir: &tempfile::TempDir, n: i64) -> DirGraph {
    let path = matches!(mode, StorageMode::Disk).then(|| dir.path());
    let mut g = new_dir_graph_in_mode(mode, path).expect("graph");
    let params = HashMap::from([("n".to_string(), Value::Int64(n))]);
    run(
        &mut g,
        "UNWIND range(1, $n) AS i CREATE (:T {id: i, title: 'T' + toString(i), a: i * 3, \
         f: toFloat(i) / 4.0, s: 'row-' + toString(i), d: date('2000-01-01') + duration({days: i}), \
         flag: i % 2 = 0, tags: [i, i + 1], \
         sparse: CASE WHEN i % 1000 = 0 THEN i ELSE null END})",
        &params,
    );
    run(
        &mut g,
        "UNWIND range(1, $n / 10) AS i CREATE (:U {id: i, title: 'U' + toString(i), w: i})",
        &params,
    );
    run(
        &mut g,
        "UNWIND range(1, $n - 1) AS i MATCH (a:T {id: i}), (b:T {id: i + 1}) \
         CREATE (a)-[:R {k: i}]->(b)",
        &params,
    );
    run(
        &mut g,
        "UNWIND range(1, $n / 10) AS i MATCH (a:T {id: i}), (u:U {id: i}) CREATE (a)-[:S]->(u)",
        &params,
    );
    run(
        &mut g,
        "MATCH (t:T) WHERE t.id % 7 = 0 SET t.s = 'rewritten-longer-' + toString(t.id)",
        &params,
    );
    run(
        &mut g,
        "MATCH (t:T {id: 5}) SET t.f = 9007199254740993",
        &params,
    );
    g
}

/// Every third node, `T` and `U` interleaved in index order.
fn kept(g: &DirGraph) -> Vec<NodeIndex> {
    g.graph
        .node_indices()
        .filter(|n| n.index() % 3 == 0)
        .collect()
}

/// The copy [`copy_nodes`] made before the gather: each node inserted on its
/// own through [`copy_node`].
fn per_node_copy(source: &DirGraph, nodes: &[NodeIndex]) -> DirGraph {
    let mut dest = DirGraph::new();
    clone_subset_metadata(&mut dest, source);
    for &node in nodes {
        copy_node(source, &mut dest, node);
    }
    dest
}

fn node_record(g: &DirGraph, idx: NodeIndex) -> (String, Value, Value, Vec<(String, Value)>) {
    let view = g.graph.node_view(idx).expect("node");
    let mut props: Vec<(String, Value)> = view
        .property_pairs()
        .into_iter()
        .filter(|(_, value)| !matches!(value, Value::Null))
        .map(|(key, value)| (g.interner.resolve(key).to_string(), value))
        .collect();
    props.sort_by(|a, b| a.0.cmp(&b.0));
    (
        view.node_type_str(&g.interner).to_string(),
        view.id().into_owned(),
        view.title().into_owned(),
        props,
    )
}

#[test]
fn the_gathered_copy_reads_back_what_the_per_node_copy_does_in_every_mode() {
    for mode in MODES {
        let dir = tempfile::tempdir().unwrap();
        let mut source = graph(mode, &dir, 3_000);
        // An auto-timestamp type: a copy keeps the source's provenance (here,
        // none) instead of stamping the copy time on either route.
        source.schema_definition = Some(SchemaDefinition {
            node_schemas: HashMap::from([(
                "T".to_string(),
                NodeSchemaDefinition {
                    auto_timestamp: Some(true),
                    ..Default::default()
                },
            )]),
            ..Default::default()
        });
        let nodes = kept(&source);
        let (copy, index_map) = copy_induced_subgraph(&source, &nodes, |_| true).unwrap();
        let reference = per_node_copy(&source, &nodes);
        assert_eq!(copy.graph.node_count(), nodes.len(), "{mode:?}");
        for (i, &old) in nodes.iter().enumerate() {
            let new = NodeIndex::new(i);
            assert_eq!(index_map[&old], new, "{mode:?}: copy order");
            let got = node_record(&copy, new);
            assert_eq!(got, node_record(&reference, new), "{mode:?}: node {i}");
            assert_eq!(got, node_record(&source, old), "{mode:?}: node {i}");
        }
        for node_type in ["T", "U"] {
            let expected: Vec<NodeIndex> = (0..nodes.len())
                .map(NodeIndex::new)
                .filter(|&n| node_record(&copy, n).0 == node_type)
                .collect();
            assert!(
                !expected.is_empty(),
                "{mode:?}: premise, {node_type} is kept"
            );
            assert!(
                copy.type_indices.get(node_type).map(|v| v.to_vec()) == Some(expected),
                "{mode:?}: {node_type} type index"
            );
            assert_eq!(
                copy.node_type_metadata.get(node_type),
                source.node_type_metadata.get(node_type),
                "{mode:?}: {node_type} metadata"
            );
        }
        // Every relationship between two kept nodes, with its properties.
        let edges = |g: &DirGraph, map: &dyn Fn(NodeIndex) -> Option<NodeIndex>| {
            let mut out: Vec<EdgeRecord> = g
                .graph
                .node_indices()
                .flat_map(|n| g.graph.edges(n).collect::<Vec<_>>())
                .filter_map(|e| {
                    Some((
                        map(e.source())?.index(),
                        map(e.target())?.index(),
                        e.weight().connection_type_str(&g.interner).to_string(),
                        e.weight().properties.to_vec(),
                    ))
                })
                .collect();
            out.sort_by(|a, b| (a.0, a.1, &a.2).cmp(&(b.0, b.1, &b.2)));
            out
        };
        let copied = edges(&copy, &|n| Some(n));
        assert!(
            !copied.is_empty(),
            "{mode:?}: premise, the copy keeps edges"
        );
        assert_eq!(
            copied,
            edges(&source, &|n| index_map.get(&n).copied()),
            "{mode:?}: relationships"
        );
    }
}

/// Columns NULL on every kept row stay known, and a column widened to
/// `Mixed` keeps its stored kind even where the kept rows would fit the
/// declared `Float64`.
#[test]
fn a_gathered_copy_keeps_all_null_columns_and_widened_kinds() {
    for mode in [StorageMode::Memory, StorageMode::Mapped] {
        let dir = tempfile::tempdir().unwrap();
        let source = graph(mode, &dir, 3_000);
        // T rows whose `sparse` is NULL, past the widened row (id 5).
        let nodes: Vec<NodeIndex> = (10..110).map(NodeIndex::new).collect();
        let (copy, _) = copy_induced_subgraph(&source, &nodes, |_| true).unwrap();
        let t = InternedKey::from_str("T");
        let (src, dst) = (
            source.graph.column_store(t).unwrap(),
            copy.graph.column_store(t).unwrap(),
        );
        let sparse = InternedKey::from_str("sparse");
        let slot = dst.slot(sparse).expect("the all-NULL column is kept") as usize;
        assert!((0..dst.row_count()).all(|row| dst.get(row, sparse).is_none()));
        assert_eq!(
            dst.column_type_str(slot),
            src.column_type_str(src.slot(sparse).unwrap() as usize)
        );
        assert!(
            copy.node_type_metadata["T"].contains_key("sparse"),
            "{mode:?}"
        );
        let f = InternedKey::from_str("f");
        assert_eq!(
            src.column_type_str(src.slot(f).unwrap() as usize),
            Some("mixed")
        );
        assert_eq!(
            dst.column_type_str(dst.slot(f).unwrap() as usize),
            Some("mixed"),
            "{mode:?}: the widened column narrowed back"
        );
    }
}

/// The copy's stores share no column with the source: a write on either side
/// leaves the other's values and heap untouched, and the copy's heap is its
/// kept rows'.
#[test]
fn a_gathered_copy_shares_no_column_with_its_source() {
    for mode in [StorageMode::Memory, StorageMode::Mapped] {
        let dir = tempfile::tempdir().unwrap();
        let mut source = graph(mode, &dir, 3_000);
        let nodes: Vec<NodeIndex> = (0..30).map(NodeIndex::new).collect();
        let (mut copy, _) = copy_induced_subgraph(&source, &nodes, |_| true).unwrap();
        let t = InternedKey::from_str("T");
        {
            let (src, dst) = (
                source.graph.column_store(t).unwrap(),
                copy.graph.column_store(t).unwrap(),
            );
            assert!(!Arc::ptr_eq(src, dst), "{mode:?}");
            let columns = |s: &ColumnStore| -> Vec<*const TypedColumn> {
                s.columns_ref()
                    .chain(s.id_column_ref())
                    .chain(s.title_column_ref())
                    .map(|c| c as *const TypedColumn)
                    .collect()
            };
            let shared: Vec<_> = columns(dst)
                .into_iter()
                .filter(|c| columns(src).contains(c))
                .collect();
            assert!(
                shared.is_empty(),
                "{mode:?}: {} shared columns",
                shared.len()
            );
            assert!(
                dst.heap_bytes() * 20 < src.heap_bytes().max(1) * 2 || src.heap_bytes() == 0,
                "{mode:?}: 30 of 3000 rows hold {} heap bytes, the source {}",
                dst.heap_bytes(),
                src.heap_bytes()
            );
        }
        let read = |g: &DirGraph, idx: usize| node_record(g, NodeIndex::new(idx)).3;
        let (source_before, source_heap) = (
            read(&source, 1),
            source.graph.column_store(t).unwrap().heap_bytes(),
        );
        let params = HashMap::new();
        run(
            &mut copy,
            "MATCH (t:T) SET t.a = -1, t.s = 'copy', t.tags = []",
            &params,
        );
        assert_eq!(
            read(&source, 1),
            source_before,
            "{mode:?}: the copy's write reached the source"
        );
        assert_eq!(
            source.graph.column_store(t).unwrap().heap_bytes(),
            source_heap,
            "{mode:?}"
        );
        let (copy_before, copy_heap) = (
            read(&copy, 1),
            copy.graph.column_store(t).unwrap().heap_bytes(),
        );
        run(
            &mut source,
            "MATCH (t:T) SET t.a = -2, t.s = 'source'",
            &params,
        );
        assert_eq!(
            read(&copy, 1),
            copy_before,
            "{mode:?}: the source's write reached the copy"
        );
        assert_eq!(
            copy.graph.column_store(t).unwrap().heap_bytes(),
            copy_heap,
            "{mode:?}"
        );
    }
}

/// A copy's heap scales with the rows it keeps, not the rows its source
/// holds.
#[test]
fn a_copy_heap_scales_with_kept_rows() {
    let dir = tempfile::tempdir().unwrap();
    let source = graph(StorageMode::Memory, &dir, 20_000);
    let t = InternedKey::from_str("T");
    let heap = |kept: usize| {
        let nodes: Vec<NodeIndex> = (0..kept).map(NodeIndex::new).collect();
        let (copy, _) = copy_induced_subgraph(&source, &nodes, |_| true).unwrap();
        copy.graph.column_store(t).unwrap().heap_bytes()
    };
    let source_heap = source.graph.column_store(t).unwrap().heap_bytes();
    let (small, large) = (heap(20), heap(2_000));
    assert!(
        small * 100 < source_heap,
        "20 rows: {small} of the source's {source_heap}"
    );
    assert!(
        large > 50 * small && large < 200 * small,
        "100x the rows took {large} heap bytes against {small}"
    );
}

/// A streaming disk-to-disk copy carries an integer title and a timestamp
/// property across intact, from a source that is still heap-backed and from one
/// that a save has already re-pointed at its mmap files. The writer takes its
/// title kind from the source store, and an integer title has no borrowed-`&str`
/// form, so a copy that reads it as a string loses every title (or refuses the
/// copy).
#[test]
fn a_streaming_copy_carries_integer_titles_and_timestamps_from_either_kind_of_source() {
    for save_first in [false, true] {
        let source_dir = tempfile::tempdir().unwrap();
        let mut source = new_dir_graph_in_mode(StorageMode::Disk, Some(source_dir.path())).unwrap();
        let base = chrono::NaiveDate::from_ymd_opt(2010, 1, 1)
            .unwrap()
            .and_hms_micro_opt(0, 0, 1, 250_000)
            .unwrap();
        let rows = (1..=40i64)
            .map(|i| {
                vec![
                    Value::Int64(i),
                    Value::Int64(7_100_000_000_000 + i),
                    Value::Timestamp(base + chrono::Duration::days(i)),
                    Value::Int64(i * 2),
                    Value::Float64(i as f64 * 0.25),
                    Value::Boolean(i % 2 == 0),
                    Value::String(format!("desk-{i}")),
                    Value::DateTime(
                        chrono::NaiveDate::from_ymd_opt(1990, 1, 1).unwrap()
                            + chrono::Duration::days(i),
                    ),
                ]
            })
            .collect();
        let frame = crate::datatypes::DataFrame::from_cypher_rows(
            [
                "id", "badge", "issued", "grade", "ratio", "active", "desk", "hired",
            ]
            .map(String::from)
            .to_vec(),
            rows,
        )
        .unwrap();
        crate::graph::mutation::maintain::add_nodes(
            &mut source,
            frame,
            "Badge".into(),
            "id".into(),
            Some("badge".into()),
            None,
        )
        .unwrap();
        if save_first {
            source
                .save_disk(source_dir.path().to_str().unwrap())
                .unwrap();
        }
        let original = source.column_store("Badge").unwrap();
        assert_eq!(original.has_mmap_base(), save_first);
        assert_eq!(original.title_type_str(), Some("int64"));

        let kept: Vec<u32> = source
            .type_indices
            .get("Badge")
            .unwrap()
            .iter()
            .map(|n| n.index() as u32)
            .collect();
        let out_dir = tempfile::tempdir().unwrap();
        let out = out_dir.path().join("copy");
        let per_type = HashMap::from([("Badge".to_string(), kept)]);
        crate::graph::mutation::subgraph_streaming::save_subset_streaming_disk(
            &source, &per_type, None, &out, None,
        )
        .unwrap();

        let copy = crate::graph::io::file::load_file(out.to_str().unwrap()).unwrap();
        let copied = copy.column_store("Badge").unwrap();
        assert_eq!(copied.row_count(), 40, "save_first={save_first}");
        assert_eq!(
            copied.title_type_str(),
            Some("int64"),
            "save_first={save_first}"
        );
        // Every property of the copy is a typed column, whichever kind of store
        // the source was: none rides in the overflow bag.
        let layout: HashMap<_, _> = copied
            .property_layout()
            .into_iter()
            .map(|(key, kind)| (copy.interner.resolve(key).to_string(), kind))
            .collect();
        for (name, kind) in [
            ("issued", "timestamp"),
            ("grade", "int64"),
            ("ratio", "float64"),
            ("active", "bool"),
            ("desk", "string"),
            ("hired", "date"),
        ] {
            assert_eq!(
                layout.get(name),
                Some(&kind),
                "save_first={save_first} {name}"
            );
        }
        assert!(
            !copied.pure_mmap_store().unwrap().has_overflow,
            "save_first={save_first}: a typed property was written to the overflow bag"
        );
        for row in 0..40u32 {
            assert_eq!(
                copied.get_title(row),
                original.get_title(row),
                "save_first={save_first} row {row} title"
            );
            for name in ["issued", "grade", "ratio", "active", "desk", "hired"] {
                let key = InternedKey::from_str(name);
                assert_eq!(
                    copied.get(row, key),
                    original.get(row, key),
                    "save_first={save_first} row {row} {name}: exact, timestamps to the microsecond"
                );
            }
        }
    }
}
