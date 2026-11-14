//! Property columns in a v9 container: which columns are written to the data
//! region, how each load policy places them, and what writes, clones,
//! corruption and a save over the loaded path do to a mapped column.

use super::mappable::SaveLayout;
use super::*;
use crate::graph::session::{execute_mut, execute_read, ExecuteOptions};
use std::collections::HashMap;

/// 40,000 rows x 8 bytes = 320 KB per 8-byte column, over the 256 KB floor.
/// `id` and `score` are scrambled so the delta-varint form (which a v9 save
/// keeps in the compressed section only when it is 4x smaller) does not win;
/// `seq` is monotonic, so it does.
const ROWS: i64 = 40_000;

fn opts() -> ExecuteOptions<'static> {
    // A leaked empty map keeps the options 'static for the whole test binary.
    let empty: &'static HashMap<String, Value> = Box::leak(Box::new(HashMap::new()));
    ExecuteOptions::eager(empty)
}

fn big_graph() -> DirGraph {
    let mut graph = DirGraph::new();
    execute_mut(
        &mut graph,
        &format!(
            "UNWIND range(1, {ROWS}) AS i CREATE (:Big {{id: (i * 7919) % 40009, \
             seq: i, title: 't' + toString(i), \
             score: (i * 7919) % 40009 * 104729, ratio: i / 7.0, label: 'row-' + toString(i) + '-ünï', \
             flag: i % 2 = 0, born: date('2020-01-01')}})"
        ),
        &opts(),
    )
    .unwrap();
    execute_mut(
        &mut graph,
        "UNWIND range(1, 50) AS i CREATE (:Small {id: i, score: i})",
        &opts(),
    )
    .unwrap();
    graph
}

fn save_to(graph: DirGraph, path: &std::path::Path, layout: SaveLayout) -> Arc<DirGraph> {
    let mut arc = Arc::new(graph);
    save_graph_with_layout(&mut arc, path.to_str().unwrap(), true, layout).unwrap();
    arc
}

fn save_to_arc(
    mut arc: Arc<DirGraph>,
    path: &std::path::Path,
    layout: SaveLayout,
) -> Arc<DirGraph> {
    save_graph_with_layout(&mut arc, path.to_str().unwrap(), true, layout).unwrap();
    arc
}

fn load_columns(path: &std::path::Path, policy: MapPolicy) -> Arc<DirGraph> {
    load_file_with(
        path.to_str().unwrap(),
        &LoadOptions::new().with_columns(policy),
    )
    .unwrap()
}

fn rows(graph: &DirGraph, query: &str) -> String {
    format!(
        "{:?}",
        execute_read(graph, query, &opts()).unwrap().result.rows
    )
}

fn shared_count(graph: &DirGraph, node_type: &str) -> usize {
    graph
        .column_store(node_type)
        .map_or(0, |store| store.shared_sections().len())
}

fn magic(path: &std::path::Path) -> [u8; 4] {
    std::fs::read(path).unwrap()[..4].try_into().unwrap()
}

fn metadata_of(bytes: &[u8]) -> (FileMetadata, u64) {
    let metadata = metadata_head::read_metadata_head(bytes, "test").unwrap();
    let end = 13 + u32::from_le_bytes(bytes[9..13].try_into().unwrap()) as usize;
    let base = data_region_base(&metadata, end, bytes.len() as u64).unwrap();
    (metadata, base)
}

const PROBE: &str = "MATCH (n:Big) RETURN count(n), sum(n.score), sum(n.ratio), \
                     sum(CASE WHEN n.flag THEN 1 ELSE 0 END), max(n.label), min(n.title), \
                     max(n.born)";

#[test]
fn large_columns_are_written_to_the_data_region_and_small_ones_stay_inline() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("g.kgl");
    save_to(big_graph(), &path, SaveLayout::Auto);
    assert_eq!(magic(&path), V9_MAGIC);

    let bytes = std::fs::read(&path).unwrap();
    let (metadata, _) = metadata_of(&bytes);
    let names: Vec<&str> = metadata.mappable.iter().map(|e| e.name.as_str()).collect();
    for expected in [
        "col:Big:score:data",
        "col:Big:score:nulls",
        "col:Big:ratio:data",
        "col:Big:label:offsets",
        "col:Big:label:bytes",
        "col:Big:label:nulls",
        "col:Big:__id__:data",
    ] {
        assert!(
            names.contains(&expected),
            "{expected} missing from {names:?}"
        );
    }
    assert!(
        names.iter().all(|n| !n.starts_with("col:Big:seq:")),
        "a monotonic int64 column is 9x smaller as deltas and stays compressed: {names:?}"
    );
    assert!(
        names.iter().all(|n| !n.starts_with("col:Small:")),
        "a 50-row type has nothing above the floor: {names:?}"
    );
    assert!(metadata.mappable.iter().all(|e| e.owner == "columns"));
    let label_bytes = metadata
        .mappable
        .iter()
        .find(|e| e.name == "col:Big:label:bytes")
        .unwrap();
    assert_ne!(label_bytes.flags & mappable::FLAG_UTF8_VALIDATED, 0);

    let compact = tmp.path().join("compact.kgl");
    save_to(big_graph(), &compact, SaveLayout::Compact);
    assert_eq!(magic(&compact), V8_MAGIC, "compact never writes v9");
}

#[test]
fn a_compact_save_keeps_columns_compressed() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("g.kgl");
    let original = save_to(big_graph(), &path, SaveLayout::Compact);
    assert_eq!(magic(&path), V8_MAGIC, "compact writes v8 bytes");
    let loaded = load_columns(&path, MapPolicy::Mapped);
    assert_eq!(shared_count(&loaded, "Big"), 0);
    assert_eq!(rows(&loaded, PROBE), rows(&original, PROBE));
}

#[test]
fn every_placement_serves_the_same_answers() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("g.kgl");
    let original = save_to(big_graph(), &path, SaveLayout::Auto);
    let expected = rows(&original, PROBE);

    let mapped = load_columns(&path, MapPolicy::Mapped);
    let auto = load_columns(&path, MapPolicy::Auto);
    let memory = load_columns(&path, MapPolicy::Memory);
    assert!(shared_count(&mapped, "Big") > 0);
    // `Auto` never maps on Windows, where a mapping blocks the save rename.
    assert_eq!(shared_count(&auto, "Big") > 0, cfg!(unix));
    assert_eq!(shared_count(&memory, "Big"), 0);
    assert_eq!(shared_count(&mapped, "Small"), 0);
    for graph in [&mapped, &auto, &memory] {
        assert_eq!(rows(graph, PROBE), expected);
        assert_eq!(
            rows(
                graph,
                "MATCH (n:Big) WHERE n.seq = 777 RETURN n.title, n.label, n.score"
            ),
            rows(
                &original,
                "MATCH (n:Big) WHERE n.seq = 777 RETURN n.title, n.label, n.score"
            ),
        );
    }

    // A v8 file loads through the compressed path under every policy.
    let v8 = tmp.path().join("v8.kgl");
    save_to(big_graph(), &v8, SaveLayout::Compact);
    for policy in [MapPolicy::Auto, MapPolicy::Mapped, MapPolicy::Memory] {
        let graph = load_columns(&v8, policy);
        assert_eq!(shared_count(&graph, "Big"), 0);
        assert_eq!(rows(&graph, PROBE), expected);
    }
}

#[test]
fn a_mapped_load_mints_no_spill_directory_and_a_decoded_one_does() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("g.kgl");
    // Every column of this type is above the floor and incompressible as
    // deltas, so nothing is left for the decode-and-spill path.
    let mut graph = DirGraph::new();
    execute_mut(
        &mut graph,
        &format!(
            "UNWIND range(1, {ROWS}) AS i CREATE (:Big {{id: (i * 7919) % 40009, \
             title: 't' + toString(i), score: (i * 7919) % 40009 * 104729, ratio: i / 7.0}})"
        ),
        &opts(),
    )
    .unwrap();
    save_to(graph, &path, SaveLayout::Auto);

    let spill_dirs =
        |graph: &DirGraph| -> Vec<std::path::PathBuf> { graph.temp_dirs.lock().unwrap().clone() };
    let mapped = load_columns(&path, MapPolicy::Mapped);
    assert!(shared_count(&mapped, "Big") > 0);
    assert!(
        spill_dirs(&mapped).iter().all(|dir| !dir.exists()),
        "mapped columns need no spill file"
    );
    let memory = load_columns(&path, MapPolicy::Memory);
    assert!(
        spill_dirs(&memory).iter().any(|dir| dir.exists()),
        "decoded columns of 256 KB spill, as a v8 load does"
    );

    // A monotonic integer column stays delta-compressed in the file, so its
    // decode still spills under every policy: the price of the smaller file.
    let path = tmp.path().join("seq.kgl");
    save_to(big_graph(), &path, SaveLayout::Auto);
    let mapped = load_columns(&path, MapPolicy::Mapped);
    assert!(spill_dirs(&mapped).iter().any(|dir| dir.exists()));
}

#[test]
fn an_unrequested_columns_policy_maps_in_every_storage_mode() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("g.kgl");
    save_to(big_graph(), &path, SaveLayout::Auto);
    let path_str = path.to_str().unwrap();
    let shared =
        |options: &LoadOptions| shared_count(&load_file_with(path_str, options).unwrap(), "Big");

    // Auto never maps on Windows, where a mapped file cannot be replaced.
    let maps = cfg!(unix);
    assert_eq!(
        shared(&LoadOptions::new()) > 0,
        maps,
        "in-memory mode maps unrequested columns (unix)"
    );
    assert_eq!(
        shared(&LoadOptions::new().with_storage(StorageMode::Mapped)) > 0,
        maps,
        "mapped storage mode maps unrequested columns (unix)"
    );
    assert_eq!(
        shared(&LoadOptions::new().with_columns(MapPolicy::Memory)),
        0,
        "an explicit Memory decodes in memory mode"
    );
    assert_eq!(
        shared(
            &LoadOptions::new()
                .with_storage(StorageMode::Mapped)
                .with_columns(MapPolicy::Memory)
        ),
        0,
        "an explicit Memory decodes in mapped mode too"
    );
}

#[test]
fn the_first_write_promotes_only_the_written_column_and_leaves_snapshots_mapped() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("g.kgl");
    save_to(big_graph(), &path, SaveLayout::Auto);
    let mut graph = load_columns(&path, MapPolicy::Mapped);
    let before = shared_count(&graph, "Big");
    let snapshot = Arc::clone(&graph);
    let expected_sum = rows(&snapshot, "MATCH (n:Big) RETURN sum(n.ratio)");

    execute_mut(
        Arc::make_mut(&mut graph),
        "MATCH (n:Big) WHERE n.seq = 5 SET n.ratio = 0.5",
        &opts(),
    )
    .unwrap();
    let after = shared_count(&graph, "Big");
    assert!(
        after < before,
        "the written column is promoted ({after} < {before})"
    );
    assert!(after > 0, "untouched columns stay mapped");
    assert_eq!(
        rows(&graph, "MATCH (n:Big) WHERE n.seq = 5 RETURN n.ratio"),
        "[[Float64(0.5)]]"
    );
    assert_eq!(
        rows(&snapshot, "MATCH (n:Big) RETURN sum(n.ratio)"),
        expected_sum
    );
    assert_eq!(
        shared_count(&snapshot, "Big"),
        before,
        "the snapshot stays mapped"
    );

    // Appends and string writes work on a promoted column.
    execute_mut(
        Arc::make_mut(&mut graph),
        "CREATE (:Big {id: 99999, seq: 99999, title: 'new', score: 1, ratio: 1.0, label: 'x', flag: true, born: date('2021-01-01')})",
        &opts(),
    )
    .unwrap();
    execute_mut(
        Arc::make_mut(&mut graph),
        "MATCH (n:Big) WHERE n.seq = 6 SET n.label = 'a much longer replacement label'",
        &opts(),
    )
    .unwrap();
    assert_eq!(
        rows(&graph, "MATCH (n:Big) WHERE n.seq = 6 RETURN n.label"),
        "[[String(\"a much longer replacement label\")]]"
    );
    assert_eq!(
        rows(&graph, "MATCH (n:Big) RETURN count(n)"),
        "[[Int64(40001)]]"
    );
}

#[test]
fn a_mapped_graph_saves_to_a_v9_file_and_over_its_own_path() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("g.kgl");
    save_to(big_graph(), &path, SaveLayout::Auto);
    let mut loaded = load_columns(&path, MapPolicy::Mapped);
    let expected = rows(&loaded, PROBE);

    save_graph_with_layout(&mut loaded, path.to_str().unwrap(), true, SaveLayout::Auto).unwrap();
    assert_eq!(magic(&path), V9_MAGIC);
    // Windows promotes before the rename; elsewhere the old inode stays mapped.
    assert_eq!(shared_count(&loaded, "Big") > 0, cfg!(unix));
    assert_eq!(rows(&loaded, PROBE), expected);

    let reloaded = load_columns(&path, MapPolicy::Mapped);
    assert_eq!(rows(&reloaded, PROBE), expected);

    // And out to the compact layout, which decodes every mapped column.
    let compact = tmp.path().join("compact.kgl");
    save_graph_with_layout(
        &mut loaded,
        compact.to_str().unwrap(),
        true,
        SaveLayout::Compact,
    )
    .unwrap();
    assert_eq!(magic(&compact), V8_MAGIC);
    assert_eq!(
        rows(&load_columns(&compact, MapPolicy::Mapped), PROBE),
        expected
    );
}

#[test]
fn promotion_before_a_windows_save_releases_every_column_mapping() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("g.kgl");
    save_to(big_graph(), &path, SaveLayout::Auto);
    let mut graph = load_columns(&path, MapPolicy::Mapped);
    let expected = rows(&graph, PROBE);
    assert!(shared_count(&graph, "Big") > 0);

    embedding_sections::release_mappings_of_destination(&mut graph, path.to_str().unwrap(), true);
    assert_eq!(shared_count(&graph, "Big"), 0);
    assert_eq!(rows(&graph, PROBE), expected);
}

fn flip_byte_in(path: &std::path::Path, section: &str, at: usize, refresh_crc: bool) {
    if refresh_crc {
        container_test_support::patch_section(path, section, |bytes| bytes[at] = 0xFF);
        return;
    }
    let mut bytes = std::fs::read(path).unwrap();
    let (metadata, base) = metadata_of(&bytes);
    let entry = metadata
        .mappable
        .iter()
        .find(|e| e.name == section)
        .unwrap();
    bytes[(base + entry.offset) as usize + at] = 0xFF;
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn a_corrupt_string_section_is_refused_at_open_and_a_numeric_one_by_verify() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("g.kgl");
    save_to(big_graph(), &path, SaveLayout::Auto);

    let numeric = tmp.path().join("numeric.kgl");
    std::fs::copy(&path, &numeric).unwrap();
    flip_byte_in(&numeric, "col:Big:ratio:data", 100, false);
    let graph = load_columns(&numeric, MapPolicy::Mapped);
    let report = verify_graph(&graph);
    let bad: Vec<_> = report
        .sections
        .iter()
        .filter(|s| s.error.is_some())
        .collect();
    assert_eq!(bad.len(), 1, "{report:?}");
    assert_eq!(bad[0].name, "col:Big:ratio:data");
    assert!(verify_graph(&load_columns(&path, MapPolicy::Mapped)).is_ok());
    let file_report = verify_file(numeric.to_str().unwrap()).unwrap();
    assert!(!file_report.is_ok());
    // A save refuses to copy the damage into a new file.
    let mut arc = graph;
    let copy = tmp.path().join("copy.kgl");
    assert!(
        save_graph_with_layout(&mut arc, copy.to_str().unwrap(), true, SaveLayout::Auto).is_err()
    );

    for policy in [MapPolicy::Mapped, MapPolicy::Memory] {
        let strings = tmp.path().join(format!("strings-{policy:?}.kgl"));
        std::fs::copy(&path, &strings).unwrap();
        flip_byte_in(&strings, "col:Big:label:bytes", 10, false);
        let error = load_file_with(
            strings.to_str().unwrap(),
            &LoadOptions::new().with_columns(policy),
        )
        .map(|_| ())
        .unwrap_err();
        assert!(error.to_string().contains("corrupt"), "{policy:?}: {error}");
    }
}

#[test]
fn invalid_utf8_with_a_valid_digest_is_refused_at_open() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("g.kgl");
    save_to(big_graph(), &path, SaveLayout::Auto);
    flip_byte_in(&path, "col:Big:label:bytes", 10, true);
    for policy in [MapPolicy::Mapped, MapPolicy::Memory] {
        let error = load_file_with(
            path.to_str().unwrap(),
            &LoadOptions::new().with_columns(policy),
        )
        .map(|_| ())
        .unwrap_err();
        assert!(error.to_string().contains("UTF-8"), "{policy:?}: {error}");
    }
}

#[test]
fn clones_of_a_mapped_column_buffer_share_the_mapping() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("g.kgl");
    save_to(big_graph(), &path, SaveLayout::Auto);
    let graph = load_columns(&path, MapPolicy::Mapped);
    let store = graph.column_store("Big").unwrap();
    let copy = (**store).clone();
    assert_eq!(copy.shared_sections().len(), store.shared_sections().len());
    let mut promoted = copy.clone();
    promoted.promote_shared();
    assert!(promoted.shared_sections().is_empty());
    assert_eq!(copy.shared_sections().len(), store.shared_sections().len());
}

#[test]
fn mapped_storage_mode_keeps_columns_shared_until_written() {
    use crate::graph::storage::mode::StorageMode;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("g.kgl");
    save_to(big_graph(), &path, SaveLayout::Auto);
    let expected = rows(&load_columns(&path, MapPolicy::Memory), PROBE);

    let mut graph = load_file_with(
        path.to_str().unwrap(),
        &LoadOptions::new()
            .with_storage(StorageMode::Mapped)
            .with_columns(MapPolicy::Mapped),
    )
    .unwrap();
    assert!(graph.graph.is_mapped());
    let before = shared_count(&graph, "Big");
    assert!(
        before > 0,
        "mapped storage mode does not copy mapped columns"
    );
    assert_eq!(rows(&graph, PROBE), expected);

    execute_mut(
        Arc::make_mut(&mut graph),
        "MATCH (n:Big) WHERE n.seq = 9 SET n.ratio = 2.5",
        &opts(),
    )
    .unwrap();
    assert!(shared_count(&graph, "Big") < before);
    assert_eq!(
        rows(&graph, "MATCH (n:Big) WHERE n.seq = 9 RETURN n.ratio"),
        "[[Float64(2.5)]]"
    );
    assert_eq!(
        rows(&graph, "MATCH (n:Big) WHERE n.seq = 10 RETURN n.ratio"),
        rows(
            &load_columns(&path, MapPolicy::Memory),
            "MATCH (n:Big) WHERE n.seq = 10 RETURN n.ratio"
        )
    );
}

/// A numeric column section decoded onto the heap is checked in the same
/// pass, since nothing re-checks the copy afterwards. A mapped one is checked
/// at first use instead (see `a_corrupt_string_section_is_refused_at_open_and_a_numeric_one_by_verify`).
#[test]
fn a_corrupt_numeric_section_is_refused_by_a_memory_load() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("g.kgl");
    save_to(big_graph(), &path, SaveLayout::Auto);
    flip_byte_in(&path, "col:Big:ratio:data", 100, false);
    let error = load_file_with(
        path.to_str().unwrap(),
        &LoadOptions::new().with_columns(MapPolicy::Memory),
    )
    .map(|_| ())
    .unwrap_err();
    assert!(error.to_string().contains("corrupt"), "{error}");
    if cfg!(unix) {
        // The default load maps it, so the damage surfaces at verify.
        let graph = load_file_with(path.to_str().unwrap(), &LoadOptions::new()).unwrap();
        assert!(!verify_graph(&graph).is_ok());
    }
}

/// A column write promotes the buffer to the heap; promoting a mapped buffer
/// that fails its CRC must refuse, leaving it mapped and unsaveable, instead
/// of laundering the damage into a freshly checksummed file.
#[test]
fn a_column_write_does_not_promote_a_corrupt_mapped_buffer() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("g.kgl");
    save_to(big_graph(), &path, SaveLayout::Auto);
    flip_byte_in(&path, "col:Big:ratio:data", 100, false);
    let mut graph = load_columns(&path, MapPolicy::Mapped);
    let before = shared_count(&graph, "Big");

    let wrote = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        execute_mut(
            Arc::make_mut(&mut graph),
            "MATCH (n:Big) WHERE n.seq = 5 SET n.ratio = 0.5",
            &opts(),
        )
        .is_err()
    }));
    // A panic (the promotion backstop) or an error both mean the write was refused.
    assert!(wrote.unwrap_or(true));
    assert_eq!(shared_count(&graph, "Big"), before, "nothing was promoted");
    let copy = tmp.path().join("copy.kgl");
    assert!(
        save_graph_with_layout(&mut graph, copy.to_str().unwrap(), true, SaveLayout::Auto).is_err()
    );
    assert!(!copy.exists());
}

/// `graph_info()['format_version']` must name the container on disk. The
/// load side used to re-stamp this build's constant (v8), so a v9 file read 8.
#[test]
fn format_version_names_the_container_the_save_wrote_and_the_load_parsed() {
    let tmp = tempfile::tempdir().unwrap();
    for (layout, name, version) in [
        (SaveLayout::Auto, "auto.kgl", 9u32),
        (SaveLayout::Compact, "compact.kgl", 8u32),
    ] {
        let path = tmp.path().join(name);
        let saved = save_to(big_graph(), &path, layout);
        assert_eq!(u32::from(magic(&path)[3]), version);
        assert_eq!(saved.save_metadata.format_version, version, "after save");
        let loaded = load_columns(&path, MapPolicy::Auto);
        assert_eq!(loaded.save_metadata.format_version, version, "after load");
        let again = tmp.path().join(format!("again-{name}"));
        let resaved = save_to_arc(loaded, &again, layout);
        assert_eq!(resaved.save_metadata.format_version, version, "resaved");
    }
}
