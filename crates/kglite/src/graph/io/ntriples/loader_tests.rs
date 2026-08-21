//! Unit tests for the N-Triples loader (`loader.rs`).
//!
//! Split out when `loader.rs` reached the source-quality file-line ceiling;
//! same `#[path]` convention as `storage/disk/graph_tests.rs`.

// Fixture literals like 3.14 are RDF test payloads, not stand-ins for PI.
#![allow(clippy::approx_constant)]

use super::super::parser::{XSD_BOOLEAN, XSD_DECIMAL, XSD_DOUBLE};
use super::*;
use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};

#[test]
fn test_parse_entity_triple() {
    let line = r#"<http://www.wikidata.org/entity/Q42> <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> ."#;
    let (subj, pred, obj) = parse_line(line).unwrap();
    assert!(matches!(subj, Subject::Entity("Q42")));
    assert!(matches!(pred, Predicate::WikidataDirect("P31")));
    assert!(matches!(obj, Object::Entity("Q5")));
}

#[test]
fn test_parse_literal_triple() {
    let line = r#"<http://www.wikidata.org/entity/Q42> <http://www.w3.org/2000/01/rdf-schema#label> "Douglas Adams"@en ."#;
    let (subj, pred, obj) = parse_line(line).unwrap();
    assert!(matches!(subj, Subject::Entity("Q42")));
    assert!(matches!(pred, Predicate::Label));
    assert!(matches!(obj, Object::LangLiteral(ref t, "en") if t == "Douglas Adams"));
}

#[test]
fn test_parse_typed_literal() {
    let line = r#"<http://www.wikidata.org/entity/Q31> <http://www.wikidata.org/prop/direct/P1082> "+11825551"^^<http://www.w3.org/2001/XMLSchema#decimal> ."#;
    let (_, pred, obj) = parse_line(line).unwrap();
    assert!(matches!(pred, Predicate::WikidataDirect("P1082")));
    assert!(matches!(obj, Object::TypedLiteral(ref t, _) if t == "+11825551"));
}

#[test]
fn test_parse_escaped_string() {
    let line = r#"<http://www.wikidata.org/entity/Q31> <http://www.wikidata.org/prop/direct/P1448> "K\u00F6nigreich Belgien"@de ."#;
    let (_, _, obj) = parse_line(line).unwrap();
    assert!(matches!(obj, Object::LangLiteral(ref t, "de") if t == "Königreich Belgien"));
}

#[test]
fn test_typed_literal_to_value() {
    assert_eq!(
        typed_literal_to_value("+11825551", XSD_DECIMAL),
        Value::Int64(11825551)
    );
    assert_eq!(
        typed_literal_to_value("3.14", XSD_DOUBLE),
        Value::Float64(3.14)
    );
    assert_eq!(
        typed_literal_to_value("true", XSD_BOOLEAN),
        Value::Boolean(true)
    );
}

#[test]
fn test_language_filter() {
    let filter = Some(HashSet::from(["en".to_string()]));
    assert!(language_matches("en", &filter));
    assert!(!language_matches("de", &filter));
    assert!(language_matches("de", &None));
}

#[test]
fn test_parse_qcode_number() {
    assert_eq!(parse_qcode_number("Q42"), Some(42));
    assert_eq!(parse_qcode_number("Q0"), Some(0));
    assert_eq!(parse_qcode_number("Q130000000"), Some(130_000_000));
    assert_eq!(parse_qcode_number("P31"), None); // not a Q-code
    assert_eq!(parse_qcode_number("Q"), None); // no number
    assert_eq!(parse_qcode_number(""), None); // empty
    assert_eq!(parse_qcode_number("Q-1"), None); // negative
}

#[test]
fn test_edge_buffer_compact_size() {
    // Verify compact edge buffer entry is much smaller than string-based
    assert_eq!(std::mem::size_of::<CompactNTripleEdge>(), 16);
    // String tuple is at least 72 bytes on stack (3 × 24 for String)
    assert!(std::mem::size_of::<(String, String, String)>() >= 72);
}

const VALID_TRIPLE: &[u8] = b"<http://www.wikidata.org/entity/Q1> <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> .\n";
const TWO_ENTITY_FIXTURE: &[u8] = b"<http://www.wikidata.org/entity/Q1> <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> .\n\
<http://www.wikidata.org/entity/Q1> <http://www.wikidata.org/prop/direct/P2> <http://www.wikidata.org/entity/Q2> .\n\
<http://www.wikidata.org/entity/Q2> <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> .\n";

fn test_config() -> NTriplesConfig {
    NTriplesConfig {
        predicates: None,
        languages: None,
        node_types: HashMap::new(),
        predicate_labels: HashMap::new(),
        max_entities: None,
        max_triples: None,
        verbose: false,
        auto_type: true,
        progress: None,
    }
}

fn load_error(graph: &mut DirGraph, path: &Path) -> String {
    match load_ntriples(graph, path.to_str().unwrap(), &test_config()) {
        Ok(_) => panic!("malformed N-Triples input loaded successfully"),
        Err(error) => error,
    }
}

#[derive(Clone)]
struct RecordingProgressSink {
    events: Arc<std::sync::Mutex<Vec<String>>>,
    cancel_on: Option<&'static str>,
}

impl RecordingProgressSink {
    fn event_name(event: &ProgressEvent<'_>) -> String {
        match event {
            ProgressEvent::Start { phase, .. } => format!("start:{phase}"),
            ProgressEvent::Update { phase, .. } => format!("update:{phase}"),
            ProgressEvent::Complete { phase, .. } => format!("complete:{phase}"),
        }
    }
}

impl ProgressSink for RecordingProgressSink {
    fn emit(&self, event: ProgressEvent<'_>) -> Result<(), Cancelled> {
        let name = Self::event_name(&event);
        self.events.lock().unwrap().push(name.clone());
        if self.cancel_on == Some(name.as_str()) {
            Err(Cancelled)
        } else {
            Ok(())
        }
    }
}

struct PoisonColumnBuildSink {
    data_dir: std::path::PathBuf,
}

impl ProgressSink for PoisonColumnBuildSink {
    fn emit(&self, event: ProgressEvent<'_>) -> Result<(), Cancelled> {
        if matches!(
            event,
            ProgressEvent::Start {
                phase: "phase1b",
                ..
            }
        ) {
            std::fs::create_dir_all(self.data_dir.join("columns.bin")).unwrap();
        }
        Ok(())
    }
}

fn graph_for_mode(mode: StorageMode, root: &Path) -> DirGraph {
    let mut graph =
        new_dir_graph_in_mode(mode, (mode == StorageMode::Disk).then_some(root)).unwrap();
    graph.spill_dir = Some(root.join("spill"));
    graph
}

fn column_boundary_fixture() -> String {
    let mut lines = String::new();
    for qid in 1..=40 {
        lines.push_str(&format!(
            "<http://www.wikidata.org/entity/Q{qid}> <http://www.wikidata.org/prop/direct/P31> <http://www.wikidata.org/entity/Q5> .\n"
        ));
        if qid <= 2 {
            lines.push_str(&format!(
                "<http://www.wikidata.org/entity/Q{qid}> <http://www.wikidata.org/prop/direct/P10> \"dense-{qid}\" .\n"
            ));
        }
        if qid == 1 {
            lines.push_str(
                "<http://www.wikidata.org/entity/Q1> <http://www.wikidata.org/prop/direct/P11> \"overflow\" .\n",
            );
            lines.push_str(
                "<http://www.wikidata.org/entity/Q1> <http://www.wikidata.org/prop/direct/P12> \"7\"^^<http://www.w3.org/2001/XMLSchema#decimal> .\n",
            );
        } else if qid == 2 {
            lines.push_str(
                "<http://www.wikidata.org/entity/Q2> <http://www.wikidata.org/prop/direct/P12> \"seven\" .\n",
            );
        }
    }
    lines
}

fn boundary_config() -> NTriplesConfig {
    let mut config = test_config();
    config
        .node_types
        .insert("Q5".to_string(), "Human".to_string());
    config
        .predicate_labels
        .insert("P10".to_string(), "dense".to_string());
    config
        .predicate_labels
        .insert("P11".to_string(), "sparse".to_string());
    config
        .predicate_labels
        .insert("P12".to_string(), "mixed".to_string());
    config
}

fn assert_column_boundary_values(graph: &DirGraph) {
    let dense = InternedKey::from_str("dense");
    let sparse = InternedKey::from_str("sparse");
    let mixed = InternedKey::from_str("mixed");
    let q1 = graph
        .lookup_by_id_normalized("Human", &Value::UniqueId(1))
        .unwrap();
    let q2 = graph
        .lookup_by_id_normalized("Human", &Value::UniqueId(2))
        .unwrap();
    let q3 = graph
        .lookup_by_id_normalized("Human", &Value::UniqueId(3))
        .unwrap();

    assert!(graph.column_store("Human").is_some());
    assert_eq!(
        graph.graph.get_node_property(q1, dense),
        Some(Value::String("dense-1".to_string()))
    );
    assert_eq!(graph.graph.get_node_property(q3, dense), None);
    assert_eq!(
        graph.graph.get_node_property(q1, sparse),
        Some(Value::String("overflow".to_string()))
    );
    assert_eq!(
        graph.graph.get_node_property(q1, mixed),
        Some(Value::Int64(7))
    );
    // Current direct-build behavior: the first non-null value fixes the
    // dense type and a later incompatible value is silently left null.
    assert_eq!(graph.graph.get_node_property(q2, mixed), None);
}

fn assert_column_layout_is_valid(data_dir: &Path) {
    let file_len = std::fs::metadata(data_dir.join("columns.bin"))
        .unwrap()
        .len() as usize;
    let metadata: Vec<ColumnTypeMeta> =
        serde_json::from_slice(&std::fs::read(data_dir.join("columns_meta.json")).unwrap())
            .unwrap();
    let human = metadata.iter().find(|m| m.type_name == "Human").unwrap();
    let dense = InternedKey::from_str("dense").as_u64();
    let sparse = InternedKey::from_str("sparse").as_u64();
    let mixed = InternedKey::from_str("mixed").as_u64();
    assert!(human.col_map.iter().any(|entry| entry.key_u64 == dense));
    assert!(human.col_map.iter().any(|entry| entry.key_u64 == mixed));
    assert!(!human.col_map.iter().any(|entry| entry.key_u64 == sparse));
    assert!(human.has_overflow);
    let mut regions = vec![
        human.id_data,
        human.id_nulls,
        human.id_str_data,
        human.id_str_offsets,
        human.title_data,
        human.title_offsets,
        human.title_nulls,
        human.overflow_offsets,
        human.overflow_data,
    ];
    for col in &human.fixed_cols {
        regions.extend([col.data, col.nulls]);
    }
    for col in &human.str_cols {
        regions.extend([col.data, col.offsets, col.nulls]);
    }
    regions.retain(|region| region.len > 0);
    regions.sort_by_key(|region| region.offset);
    for region in &regions {
        assert!(region.offset.checked_add(region.len).unwrap() <= file_len);
    }
    for pair in regions.windows(2) {
        assert!(pair[0].offset + pair[0].len <= pair[1].offset);
    }
}

#[test]
fn empty_input_is_valid_in_every_storage_mode() {
    let fixture = tempfile::NamedTempFile::new().unwrap();
    for mode in [StorageMode::Memory, StorageMode::Mapped, StorageMode::Disk] {
        let root = tempfile::tempdir().unwrap();
        let mut graph = graph_for_mode(mode, root.path());
        let stats =
            load_ntriples(&mut graph, fixture.path().to_str().unwrap(), &test_config()).unwrap();
        assert_eq!(stats.entities_created, 0);
        assert_eq!(stats.edges_created, 0);
        assert_eq!(graph.graph.node_count(), 0);
    }
}

#[test]
fn column_builder_boundaries_and_disk_reopen_are_characterized() {
    let fixture = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(fixture.path(), column_boundary_fixture()).unwrap();

    let mapped_root = tempfile::tempdir().unwrap();
    let mut mapped = graph_for_mode(StorageMode::Mapped, mapped_root.path());
    load_ntriples(
        &mut mapped,
        fixture.path().to_str().unwrap(),
        &boundary_config(),
    )
    .unwrap();
    assert_column_boundary_values(&mapped);
    assert_column_layout_is_valid(mapped.spill_dir.as_ref().unwrap());

    let disk_root = tempfile::tempdir().unwrap();
    let mut disk = graph_for_mode(StorageMode::Disk, disk_root.path());
    load_ntriples(
        &mut disk,
        fixture.path().to_str().unwrap(),
        &boundary_config(),
    )
    .unwrap();
    assert_column_boundary_values(&disk);
    assert_column_layout_is_valid(&disk_root.path().join("seg_000"));
    drop(disk);

    let reopened = crate::graph::io::file::load_file(disk_root.path().to_str().unwrap()).unwrap();
    assert_column_boundary_values(&reopened);
}

/// A disk build published at the graph root is what a reload sees, even
/// when the directory already carried a published generation.
///
/// Both halves matter. Creation publishes an empty generation so a crash
/// before the first `save()` leaves a loadable path, and `resolve_snapshot`
/// prefers a `CURRENT` pointer over the flat root — so without retiring the
/// pointer, this reload would come back empty while reporting success. And
/// pointing a fresh disk create at a directory that already holds a saved
/// graph is a *rebuild*: the build the caller just ran is the graph, not
/// the snapshot it replaced.
#[test]
fn a_disk_build_supersedes_a_previously_published_generation() {
    use crate::graph::storage::mode::{new_dir_graph_in_mode, StorageMode};

    let fixture = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(fixture.path(), column_boundary_fixture()).unwrap();
    let root = tempfile::tempdir().unwrap();

    // A saved generation to build over.
    let mut published = new_dir_graph_in_mode(StorageMode::Disk, Some(root.path())).unwrap();
    published
        .save_disk(root.path().to_str().unwrap())
        .expect("publish a generation");
    drop(published);
    assert!(root.path().join("CURRENT").is_file());

    let mut rebuilt = graph_for_mode(StorageMode::Disk, root.path());
    load_ntriples(
        &mut rebuilt,
        fixture.path().to_str().unwrap(),
        &boundary_config(),
    )
    .unwrap();
    drop(rebuilt);

    let reopened = crate::graph::io::file::load_file(root.path().to_str().unwrap()).unwrap();
    assert_column_boundary_values(&reopened);
}

/// A build run on a handle that already published a generation lands in a
/// *new* generation, and the old one is untouched.
///
/// After `save_disk`, the handle's `data_dir` is the published snapshot's
/// `seg_000/`, so a build that writes to `active_write_dir()` writes into
/// an immutable, reader-visible generation — the fresh `interner.json`
/// landing beside the snapshot's `interner.bin.zst`, which shadows it on
/// reload ("directory contains an unresolved type key").
#[test]
fn a_disk_build_on_a_saved_handle_publishes_a_new_generation() {
    let fixture = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(fixture.path(), column_boundary_fixture()).unwrap();
    let root = tempfile::tempdir().unwrap();

    let mut graph = graph_for_mode(StorageMode::Disk, root.path());
    graph
        .save_disk(root.path().to_str().unwrap())
        .expect("publish a generation");
    let published = crate::graph::storage::disk::generation::resolve_snapshot(root.path())
        .unwrap()
        .snapshot_dir;
    let before = digest_tree(&published);

    load_ntriples(
        &mut graph,
        fixture.path().to_str().unwrap(),
        &boundary_config(),
    )
    .unwrap();
    drop(graph);

    assert_eq!(
        digest_tree(&published),
        before,
        "a published generation is immutable"
    );
    let reopened = crate::graph::io::file::load_file(root.path().to_str().unwrap()).unwrap();
    assert_column_boundary_values(&reopened);
}

/// A build on a graph *opened from* a saved directory publishes a new
/// generation too. The handle never saved anything itself, but its write
/// target is the snapshot it was loaded from — equally immutable.
#[test]
fn a_disk_build_on_a_loaded_graph_publishes_a_new_generation() {
    let fixture = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(fixture.path(), column_boundary_fixture()).unwrap();
    let root = tempfile::tempdir().unwrap();

    let mut seeded = graph_for_mode(StorageMode::Disk, root.path());
    seeded.save_disk(root.path().to_str().unwrap()).unwrap();
    drop(seeded);

    let published = crate::graph::storage::disk::generation::resolve_snapshot(root.path())
        .unwrap()
        .snapshot_dir;
    let before = digest_tree(&published);

    let mut loaded = crate::graph::io::file::load_file(root.path().to_str().unwrap()).unwrap();
    let graph = crate::graph::handle::make_dir_graph_mut(&mut loaded);
    graph.spill_dir = Some(root.path().join("spill-loaded"));
    load_ntriples(graph, fixture.path().to_str().unwrap(), &boundary_config()).unwrap();
    drop(loaded);

    assert_eq!(
        digest_tree(&published),
        before,
        "the generation the graph was loaded from is immutable"
    );
    let reopened = crate::graph::io::file::load_file(root.path().to_str().unwrap()).unwrap();
    assert_column_boundary_values(&reopened);
}

/// A build that follows any other mutation reaches the directory.
///
/// The first mutation stages the handle in a mutation workspace, and the
/// workspace is removed when the handle drops. A build that finalised
/// there published its metadata into a directory nobody resolves and then
/// deleted the lot: `load_ntriples` reported success and the reload
/// returned the pre-build graph.
#[test]
fn a_disk_build_after_a_mutation_is_not_lost_with_the_workspace() {
    let fixture = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(fixture.path(), column_boundary_fixture()).unwrap();
    let root = tempfile::tempdir().unwrap();

    let mut graph = graph_for_mode(StorageMode::Disk, root.path());
    graph.prepare_disk_mutation().unwrap();
    load_ntriples(
        &mut graph,
        fixture.path().to_str().unwrap(),
        &boundary_config(),
    )
    .unwrap();
    drop(graph);

    let reopened = crate::graph::io::file::load_file(root.path().to_str().unwrap()).unwrap();
    assert_column_boundary_values(&reopened);
}

/// An explicit `save()` after an ntriples disk build keeps the build's
/// property columns.
///
/// The build's stores are mmap-backed (`columns.bin`), which the unified
/// column writer cannot plan, so the save rewrites every type as a
/// `columns.zst` sidecar. The sidecar carries the data, but the reload
/// rebuilds `type_schemas` from `node_type_metadata` — which this build
/// path never writes — and every column whose name the schema did not
/// know used to be dropped on the floor: the graph reloaded with its
/// nodes, ids and titles intact and every property `null`.
#[test]
fn a_save_after_a_disk_build_keeps_its_property_columns() {
    let fixture = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(fixture.path(), column_boundary_fixture()).unwrap();
    let root = tempfile::tempdir().unwrap();

    let mut graph = graph_for_mode(StorageMode::Disk, root.path());
    load_ntriples(
        &mut graph,
        fixture.path().to_str().unwrap(),
        &boundary_config(),
    )
    .unwrap();
    graph.save_disk(root.path().to_str().unwrap()).unwrap();
    drop(graph);

    let reopened = crate::graph::io::file::load_file(root.path().to_str().unwrap()).unwrap();
    assert_column_boundary_values(&reopened);
}

/// Sorted (relative path, length, bytes-hash) of every file under `dir`.
fn digest_tree(dir: &Path) -> Vec<(String, u64, u64)> {
    let mut out = Vec::new();
    for entry in walkdir::WalkDir::new(dir).follow_links(false) {
        let entry = entry.unwrap();
        if !entry.file_type().is_file() {
            continue;
        }
        let bytes = std::fs::read(entry.path()).unwrap();
        let mut hash = 1469598103934665603u64;
        for byte in &bytes {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
        out.push((
            entry
                .path()
                .strip_prefix(dir)
                .unwrap()
                .display()
                .to_string(),
            bytes.len() as u64,
            hash,
        ));
    }
    out.sort();
    out
}

#[test]
fn progress_phase_order_is_stable_across_storage_modes() {
    let fixture = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(fixture.path(), TWO_ENTITY_FIXTURE).unwrap();

    for (mode, expected) in [
        (
            StorageMode::Memory,
            vec![
                "start:phase1",
                "complete:phase1",
                "start:phase2",
                "complete:phase2",
            ],
        ),
        (
            StorageMode::Mapped,
            vec![
                "start:phase1",
                "complete:phase1",
                "start:phase1b",
                "update:phase1b",
                "complete:phase1b",
                "start:phase2",
                "complete:phase2",
            ],
        ),
        (
            StorageMode::Disk,
            vec![
                "start:phase1",
                "complete:phase1",
                "start:phase1b",
                "update:phase1b",
                "complete:phase1b",
                "start:phase2",
                "complete:phase2",
                "start:phase3",
                "complete:phase3",
                "start:finalising",
                "complete:finalising",
            ],
        ),
    ] {
        let root = tempfile::tempdir().unwrap();
        let mut graph = graph_for_mode(mode, root.path());
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut config = test_config();
        config.progress = Some(Box::new(RecordingProgressSink {
            events: Arc::clone(&events),
            cancel_on: None,
        }));

        load_ntriples(&mut graph, fixture.path().to_str().unwrap(), &config).unwrap();
        assert_eq!(*events.lock().unwrap(), expected, "mode={mode:?}");
    }
}

#[test]
fn phase2_start_cancellation_retains_nodes_but_not_edges_or_completion_marker() {
    let fixture = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(fixture.path(), TWO_ENTITY_FIXTURE).unwrap();

    for mode in [StorageMode::Memory, StorageMode::Mapped, StorageMode::Disk] {
        let root = tempfile::tempdir().unwrap();
        let mut graph = graph_for_mode(mode, root.path());
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut config = test_config();
        config.progress = Some(Box::new(RecordingProgressSink {
            events,
            cancel_on: Some("start:phase2"),
        }));

        let error = match load_ntriples(&mut graph, fixture.path().to_str().unwrap(), &config) {
            Ok(_) => panic!("mode={mode:?} ignored phase2 cancellation"),
            Err(error) => error,
        };
        assert_eq!(error, CANCELLED_TOKEN, "mode={mode:?}");
        assert_eq!(graph.graph.node_count(), 2, "mode={mode:?}");
        assert_eq!(graph.graph.edge_count(), 0, "mode={mode:?}");
        if mode == StorageMode::Disk {
            assert!(!root.path().join("metadata.json").exists());
        }
    }
}

#[test]
fn phase1b_update_cancellation_stops_before_edges_or_completion_publish() {
    let fixture = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(fixture.path(), TWO_ENTITY_FIXTURE).unwrap();

    for mode in [StorageMode::Mapped, StorageMode::Disk] {
        let root = tempfile::tempdir().unwrap();
        let mut graph = graph_for_mode(mode, root.path());
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut config = test_config();
        config.progress = Some(Box::new(RecordingProgressSink {
            events: Arc::clone(&events),
            cancel_on: Some("update:phase1b"),
        }));

        let error = match load_ntriples(&mut graph, fixture.path().to_str().unwrap(), &config) {
            Ok(_) => panic!("mode={mode:?} ignored phase1b cancellation"),
            Err(error) => error,
        };
        assert_eq!(error, CANCELLED_TOKEN, "mode={mode:?}");
        assert_eq!(
            *events.lock().unwrap(),
            [
                "start:phase1",
                "complete:phase1",
                "start:phase1b",
                "update:phase1b",
            ],
            "mode={mode:?}"
        );
        assert_eq!(graph.graph.node_count(), 2);
        assert_eq!(graph.graph.edge_count(), 0);
        if mode == StorageMode::Disk {
            assert!(!root.path().join("metadata.json").exists());
        }
    }
}

#[test]
fn phase1b_io_failure_keeps_its_io_classification() {
    let fixture = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(fixture.path(), TWO_ENTITY_FIXTURE).unwrap();
    let root = tempfile::tempdir().unwrap();
    let mut graph = graph_for_mode(StorageMode::Mapped, root.path());
    let mut config = test_config();
    config.progress = Some(Box::new(PoisonColumnBuildSink {
        data_dir: graph.spill_dir.clone().unwrap(),
    }));

    let error = match load_ntriples(&mut graph, fixture.path().to_str().unwrap(), &config) {
        Ok(_) => panic!("poisoned columns.bin path must fail"),
        Err(error) => error,
    };
    assert!(error.starts_with("Failed to build columns: "), "{error}");
    assert_ne!(error, CANCELLED_TOKEN);
}

struct ErrorAfterData {
    data: std::io::Cursor<Vec<u8>>,
}

impl Read for ErrorAfterData {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let read = self.data.read(buf)?;
        if read == 0 {
            Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "injected reader failure",
            ))
        } else {
            Ok(read)
        }
    }
}

struct PanicReader;

impl Read for PanicReader {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        panic!("injected reader panic")
    }
}

struct PoisonFinalisationSink {
    root_dir: std::path::PathBuf,
    completed: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl ProgressSink for PoisonFinalisationSink {
    fn emit(&self, event: ProgressEvent<'_>) -> Result<(), Cancelled> {
        match event {
            ProgressEvent::Start {
                phase: "finalising",
                ..
            } => {
                // The DirGraph sidecars are published at the graph ROOT
                // (next to disk_graph_meta.json) — poison the interner
                // path there.
                std::fs::create_dir(self.root_dir.join("interner.json")).unwrap();
            }
            ProgressEvent::Complete { phase, .. } => {
                self.completed.lock().unwrap().push(phase.to_string());
            }
            _ => {}
        }
        Ok(())
    }
}

#[test]
fn finalisation_write_failure_is_reported_before_complete_or_metadata_publish() {
    let fixture = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(fixture.path(), VALID_TRIPLE).unwrap();
    let mut graph = DirGraph::new();
    graph.enable_disk_mode().unwrap();
    let root_dir = match &graph.graph {
        crate::graph::schema::GraphBackend::Disk(disk) => disk
            .data_dir
            .parent()
            .expect("segment dir has a graph root")
            .to_path_buf(),
        _ => unreachable!(),
    };
    let completed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut config = test_config();
    config.progress = Some(Box::new(PoisonFinalisationSink {
        root_dir: root_dir.clone(),
        completed: std::sync::Arc::clone(&completed),
    }));

    let error = match load_ntriples(&mut graph, fixture.path().to_str().unwrap(), &config) {
        Ok(_) => panic!("poisoned interner output must fail finalisation"),
        Err(error) => error,
    };
    assert!(error.contains("Failed to write interner"), "{error}");
    assert!(!completed.lock().unwrap().iter().any(|p| p == "finalising"));
    assert!(
        !root_dir.join("metadata.json").exists(),
        "root completion metadata must be withheld on finalisation failure"
    );
}

#[test]
fn reader_error_is_ordered_after_prior_batches_and_propagated() {
    let reader = ErrorAfterData {
        data: std::io::Cursor::new(VALID_TRIPLE.to_vec()),
    };
    let (rx, handle) = spawn_reader(Box::new(reader));
    let messages: Vec<_> = rx.into_iter().collect();
    assert_eq!(messages.len(), 2);
    assert!(messages[0]
        .as_ref()
        .is_ok_and(|batch| batch.offsets.len() == 1));
    assert!(messages[1]
        .as_ref()
        .is_err_and(|error| error.contains("injected reader failure")));
    assert!(join_reader(handle)
        .unwrap_err()
        .contains("injected reader failure"));
}

#[test]
fn reader_thread_panic_is_propagated() {
    let (rx, handle) = spawn_reader(Box::new(PanicReader));
    drop(rx);
    assert!(join_reader(handle)
        .unwrap_err()
        .contains("injected reader panic"));
}

#[test]
fn accepted_line_with_invalid_utf8_is_rejected() {
    let subject = VALID_TRIPLE.iter().position(|byte| *byte == b'Q').unwrap() + 1;
    let predicate = VALID_TRIPLE
        .windows(2)
        .position(|window| window == b"P3")
        .unwrap()
        + 1;
    let object = VALID_TRIPLE.iter().rposition(|byte| *byte == b'Q').unwrap() + 1;

    for corrupt_at in [subject, predicate, object] {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let mut line = VALID_TRIPLE.to_vec();
        line.insert(corrupt_at, 0xff);
        std::fs::write(temp.path(), line).unwrap();
        let mut graph = DirGraph::new();
        let error = load_error(&mut graph, temp.path());
        assert!(error.contains("invalid UTF-8"));
    }
}

#[test]
fn truncated_gzip_after_valid_triple_is_not_clean_eof() {
    use std::io::Write as _;

    let temp = tempfile::Builder::new().suffix(".gz").tempfile().unwrap();
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(VALID_TRIPLE).unwrap();
    let mut compressed = encoder.finish().unwrap();
    compressed.truncate(compressed.len() - 6);
    std::fs::write(temp.path(), compressed).unwrap();

    let mut graph = DirGraph::new();
    let error = load_error(&mut graph, temp.path());
    assert!(error.contains("reader error"), "{error}");
}

#[test]
fn truncated_bzip2_after_valid_triple_is_not_clean_eof() {
    use std::io::Write as _;

    let temp = tempfile::Builder::new().suffix(".bz2").tempfile().unwrap();
    let mut encoder = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::default());
    encoder.write_all(VALID_TRIPLE).unwrap();
    let mut compressed = encoder.finish().unwrap();
    compressed.truncate(compressed.len() - 6);
    std::fs::write(temp.path(), compressed).unwrap();

    let mut graph = DirGraph::new();
    let error = load_error(&mut graph, temp.path());
    assert!(
        error.contains("reader error")
            || (error.contains("Cannot open") && error.contains("invalid bzip2 format")),
        "{error}"
    );
}
